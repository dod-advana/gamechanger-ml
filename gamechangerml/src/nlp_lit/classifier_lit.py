"""
usage: python classifier_lit.py [-h] [--absl_flags ABSL_FLAGS] --model_path
                                MODEL_PATH --data_path DATA_PATH --num_labels
                                NUM_LABELS [--batch_size BATCH_SIZE]
                                [--max_seq_len MAX_SEQ_LEN] [--port PORT]

Start the LIT server

optional arguments:
  -h, --help            show this help message and exit
  --absl_flags ABSL_FLAGS
                        absl flags - use the default
  --model_path MODEL_PATH
                        directory of the pytorch model
  --data_path DATA_PATH
                        path + file.csv, for input data .csv
  --num_labels NUM_LABELS
                        number of labels in the classification model
  --batch_size BATCH_SIZE
                        batch size, default 8
  --max_seq_len MAX_SEQ_LEN
                        maximum sequence length up to 512, default 128
  --port PORT           LIT server port, default 5432
"""
import os
import re

import torch
import transformers as trf
from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils

from gamechangerml.src.nlp_lit.gc_dataset import GCDataset

# NOTE: additional flags defined in server_flags.py
FLAGS = flags.FLAGS


def _from_pretrained(cls, *args, **kw):
    """
    Load a transformers model in PyTorch.
    """
    try:
        return cls.from_pretrained(*args, **kw)
    except OSError as e:
        raise e


class TextClassifier(lit_model.Model):
    compute_grads: bool = True

    def __init__(self, model_name_or_path, num_labels):
        """
        Instantiate LIT.

        Args:
            model_name_or_path (str): directory containing the `pytorch`
                model

            num_labels (int): number of classification labels
        """
        self.LABELS = [str(lbl) for lbl in range(num_labels)]
        self.tokenizer = trf.AutoTokenizer.from_pretrained(model_name_or_path)
        model_config = trf.AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.model = _from_pretrained(
            trf.AutoModelForSequenceClassification,
            model_name_or_path,
            config=model_config,
        )
        self.model.eval()
        logging.info("model loaded")

    # LIT API implementation
    def max_minibatch_size(self):
        return FLAGS.batch_size

    def predict_minibatch(self, inputs):
        encoded_input = self.tokenizer.batch_encode_plus(
            [ex["sentence"] for ex in inputs],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=FLAGS.max_seq_len,
            padding="longest",
            truncation="longest_first",
        )

        if torch.cuda.is_available():
            self.model.cuda()
            for tensor in encoded_input:
                encoded_input[tensor] = encoded_input[tensor].cuda()

        # Run a forward pass with gradient.
        with torch.set_grad_enabled(self.compute_grads):
            out: trf.modeling_outputs.SequenceClassifierOutput = self.model(
                **encoded_input
            )

        # Post-process outputs.
        batched_outputs = {
            "probas": torch.nn.functional.softmax(out.logits, dim=-1),
            "input_ids": encoded_input["input_ids"],
            "ntok": torch.sum(encoded_input["attention_mask"], dim=1),
            "cls_emb": out.hidden_states[-1][:, 0],  # last layer, first token
        }
        # Add attention layers to batched_outputs
        assert len(out.attentions) == self.model.config.num_hidden_layers
        for i, layer_attention in enumerate(out.attentions):
            batched_outputs[f"layer_{i}/attention"] = layer_attention

        # Request gradients after the forward pass. Note: hidden_states[0]
        # includes position and segment encodings, as well as sub-word
        # embeddings.
        if self.compute_grads:
            scalar_pred_for_gradients = torch.max(
                batched_outputs["probas"], dim=1, keepdim=False, out=None
            )[0]

            batched_outputs["input_emb_grad"] = torch.autograd.grad(
                scalar_pred_for_gradients,
                out.hidden_states[0],
                grad_outputs=torch.ones_like(scalar_pred_for_gradients),
            )[0]

        # Return as numpy for further processing. NB: cannot use
        # v.cpu().numpy() when gradients are computed.
        detached_outputs = {
            k: v.cpu().detach().numpy() for k, v in batched_outputs.items()
        }

        # un-batch outputs so we get one record per input example.
        for output in utils.unbatch_preds(detached_outputs):
            ntok = output.pop("ntok")
            output["tokens"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("input_ids")[1 : ntok - 1]
            )
            # set token gradients - exclude special tokens
            if self.compute_grads:
                output["token_grad_sentence"] = output["input_emb_grad"][
                    1 : ntok - 1
                ]

            # Process attention.
            for key in output:
                if not re.match(r"layer_(\d+)/attention", key):
                    continue
                # Select only real tokens, since most of this matrix is padding
                output[key] = output[key][:, :ntok, :ntok].transpose((0, 2, 1))
                # Make a copy of this array to avoid memory leaks, since NumPy
                # otherwise keeps a pointer around that prevents the source
                # array from being GC'd.
                output[key] = output[key].copy()
            yield output

    def input_spec(self) -> lit_types.Spec:
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(
                vocab=self.LABELS, required=False
            ),
        }

    def output_spec(self) -> lit_types.Spec:
        ret = {
            "tokens": lit_types.Tokens(),
            "probas": lit_types.MulticlassPreds(
                parent="label", vocab=self.LABELS
            ),
            "cls_emb": lit_types.Embeddings(),
        }
        # Gradients, if requested.
        if self.compute_grads:
            ret["token_grad_sentence"] = lit_types.TokenGradients(
                align="tokens"
            )

        # Attention heads, one field for each layer.
        for i in range(self.model.config.num_hidden_layers):
            ret[f"layer_{i}/attention"] = lit_types.AttentionHeads(
                align_in="tokens", align_out="tokens"
            )
        return ret

    def spec(self):
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS),
            "src": lit_types.TextSegment(),
        }

    def fit_transform_with_metadata(self, indexed_inputs):
        raise NotImplementedError

    def get_embedding_table(self):
        raise NotImplementedError


def main(_):
    data_path_ = FLAGS.data_path
    model_path = FLAGS.model_path
    num_labels = FLAGS.num_labels

    model_path = trf.file_utils.cached_path(
        model_path, extract_compressed_file=False
    )
    if not os.path.isfile(data_path_):
        raise FileNotFoundError(data_path_)

    # Load the model we defined above.
    models = {"classifier": TextClassifier(model_path, num_labels)}
    # GC data
    datasets = {"gc-data": GCDataset(data_path_, num_labels)}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()


if __name__ == "__main__":
    from argparse import ArgumentParser

    log_fmt = "[%(asctime)s%(levelname)8s], [%(filename)s:%(lineno)s "
    log_fmt += "- %(funcName)s()], %(message)s"
    logger = logging.get_absl_logger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser(
        prog="python classifier_lit.py", description="Start the LIT server"
    )
    parser.add_argument(
        "--absl_flags",
        action="append",
        default=[],
        help="absl flags - use the default",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        required=True,
        help="directory of the pytorch model",
    )
    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        required=True,
        help="path + file.csv, for input data .csv",
    )
    parser.add_argument(
        "--num_labels",
        dest="num_labels",
        type=int,
        required=True,
        help="number of labels in the classification model",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=str,
        required=False,
        default=8,
        help="batch size, default 8",
    )
    parser.add_argument(
        "--max_seq_len",
        dest="max_seq_len",
        type=int,
        required=False,
        default=128,
        help="maximum sequence length up to 512, default 128",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=5432,
        help="LIT server port default, 5432",
    )

    args_ = parser.parse_args()
    flags.DEFINE_string("model_path", args_.model_path, "saved model")
    flags.DEFINE_string("data_path", args_.data_path, "validation data")
    flags.DEFINE_integer("batch_size", args_.batch_size, "batch size")
    flags.DEFINE_integer("max_seq_len", args_.max_seq_len, "max seq length")
    flags.DEFINE_integer("num_labels", args_.num_labels, "number of labels")
    flags.port = args_.port

    app.run(main)

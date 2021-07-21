"""
usage: python ner_training_data.py [-h] -s SENT_CSV -e ENTITY_CSV -o
                                   OUTPUT_TXT [-n N_SAMPLES] [-r]
                                   [-p {tab,space}]

Create NER training data in CoNLL format

optional arguments:
  -h, --help            show this help message and exit
  -s SENT_CSV, --sentence-csv SENT_CSV
                        csv of sentences and labels
  -e ENTITY_CSV, --entity-csv ENTITY_CSV
                        csv of entities & types
  -o OUTPUT_TXT, --output-txt OUTPUT_TXT
                        output file in CoNLL-2003 format
  -n N_SAMPLES, --n-samples N_SAMPLES
                        how many samples to extract and tag (0 means get
                        everything)
  -r, --shuffle         randomly shuffle the sentence data
  -p {tab,space}, --separator {tab,space}
                        token <-> label separator, default is 'space'
"""
import logging
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import gamechangerml.src.entity.entity_mentions as em
import gamechangerml.src.text_classif.utils.classifier_utils as cu

logger = logging.getLogger(__name__)


def _wc(txt):
    return txt.count(" ") + 1


def _gen_ner_training_data(abrv_re, ent_re, entity2type, sent_dict, nlp):
    SENT = "sentence"
    I_PRFX = "I-"
    B_PRFX = "B-"
    OH = "O"
    for row in sent_dict:
        sentence_text = row[SENT]
        if not sentence_text.strip():
            continue

        doc = nlp(sentence_text)
        starts_ends = [(t.idx, t.idx + len(t.orth_) - 1) for t in doc]
        ner_labels = [OH] * len(starts_ends)
        tokens = [t.orth_ for t in doc]
        ent_spans = em.entities_spans(sentence_text, ent_re, abrv_re)

        # find token indices of an extracted entity using their spans,
        # create CoNLL tags
        for ent, ent_st_end in ent_spans:
            token_idxs = [
                idx
                for idx, tkn_st_end in enumerate(starts_ends)
                if tkn_st_end[0] >= ent_st_end[0]
                and tkn_st_end[1] <= ent_st_end[1] - 1
            ]
            if _wc(ent) == 1:
                ner_labels[token_idxs[0]] = I_PRFX + entity2type[ent.lower()]
                continue

            ner_labels[token_idxs[0]] = B_PRFX + entity2type[ent.lower()]
            for idx in token_idxs[1:]:
                if ent.lower() in entity2type:
                    ner_labels[idx] = I_PRFX + entity2type[ent.lower()]
                else:
                    logger.error("KeyError: {}".format(ent.lower()))
        uniq_labels = set(ner_labels)
        yield zip(tokens, ner_labels), uniq_labels


def ner_training_data(
    entity_csv,
    sentence_csv,
    n_samples,
    nlp,
    sep,
    out_fp,
    shuffle,
):
    """
    Create NER training data in CoNLL-2003 format. For more information on
    the tagging conventions, see https://huggingface.co/datasets/conll2003

    Args:
        entity_csv (str): name of the .csv file holding entities & types

        sentence_csv (str): name of the sentence .csv

        n_samples (int): if > 0, use everything; else up to this value

        nlp (spacy.lang.en.English): spaCy language model

        sep (str): separator between entity & label

        out_fp (str): where to write the resulting `.tsv` file

        shuffle (bool): if True, randomize the order of the sentences

    """
    if sep == "space":
        SEP = " "
    elif sep == "tab":
        SEP = "\t"
    else:
        msg = "unrecognized value for `sep`, got {} ".format(sep)
        msg += "using space separator"
        logger.warning("unrecognized value for `sep`, got {}".format(sep))
        SEP = " "
    NL = "\n"
    EMPTYSTR = ""
    print_str = EMPTYSTR
    write_interval = 1024 * 10

    abrv_re, ent_re, entity2type = em.make_entity_re(entity_csv)
    sent_dict = cu.load_data(sentence_csv, n_samples, shuffle=shuffle)

    training_generator = _gen_ner_training_data(
        abrv_re, ent_re, entity2type, sent_dict, nlp
    )
    labels = set()
    count = 0
    with open(out_fp, "w") as fp:
        for zipped, uniq_labels in tqdm(
            training_generator, total=len(sent_dict), desc="sentence"
        ):
            labels = labels.union(uniq_labels)
            count += 1
            print_str += (
                NL.join([str(e[0]) + SEP + str(e[1]) for e in zipped])
                + NL
                + NL
            )
            if count > 0 and count % write_interval == 0:
                fp.write(print_str)
                count = 0
                print_str = EMPTYSTR
        if print_str:
            fp.write(print_str[:-1])

    # unique labels to a separate file
    label_str = NL.join(sorted(list(labels)))
    label_fp, ltype = os.path.splitext(out_fp)
    label_fp = label_fp + "_labels" + ltype

    with open(label_fp, "w") as fp:
        fp.write(label_str)
    logger.info("training output written to : {}".format(out_fp))
    logger.info("         labels written to : {}".format(label_fp))


def main(
    entity_csv,
    sentence_csv,
    n_samples,
    nlp,
    sep,
    shuffle,
    t_split,
):
    in_path, _ = os.path.split(sentence_csv)
    sent_names = (
        os.path.join(in_path, "train_sent.csv"),
        os.path.join(in_path, "dev_sent.csv"),
        os.path.join(in_path, "val_sent.csv"),
    )
    output_names = [p.replace("_sent.csv", ".txt.tmp") for p in sent_names]

    dev_frac = t_split + (1.0 - t_split) / 2
    df = pd.read_csv(sentence_csv, delimiter=",", header=None)
    train, dev, val = np.split(
        df.sample(frac=1),
        [int(t_split * len(df)), int(dev_frac * len(df))],
    )
    for idx, df in enumerate((train, dev, val)):
        df.to_csv(sent_names[idx], header=False, index=False, sep=",")
        logger.info("samples : {:>4,d}".format(len(df)))

    for idx, sent_csv in enumerate(sent_names):
        ner_training_data(
            entity_csv,
            sent_csv,
            n_samples,
            nlp,
            sep,
            output_names[idx],
            shuffle,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    import gamechangerml.src.text_classif.utils.log_init as li
    from gamechangerml.src.utilities.spacy_model import get_lg_nlp

    li.initialize_logger(to_file=False, log_name="none")

    fp_ = "python " + os.path.split(__file__)[-1]
    parser = ArgumentParser(
        prog=fp_,
        description="Create NER training data in CoNLL format",
    )
    parser.add_argument(
        "-s",
        "--sentence-csv",
        dest="sent_csv",
        type=str,
        help="csv of sentences and labels",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--entity-csv",
        dest="entity_csv",
        type=str,
        help="csv of entities & types",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-txt",
        dest="output_txt",
        type=str,
        required=True,
        help="output file in CoNLL-2003 format",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        dest="n_samples",
        type=int,
        required=False,
        default=0,
        help="how many samples to extract and tag (0 means get everything)",
    )
    parser.add_argument(
        "-r",
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=False,
        help="randomly shuffle the sentence data",
    )
    parser.add_argument(
        "-p",
        "--separator",
        dest="sep",
        type=str,
        choices=["tab", "space"],
        default="space",
        help="token <-> label separator, default is 'space'",
    )
    parser.add_argument(
        "-x",
        "--train-split",
        dest="t_split",
        type=float,
        default=0.80,
        help="training split",
    )
    args = parser.parse_args()

    logger.info("retrieving spaCy model")
    nlp_ = get_lg_nlp()
    logger.info("spaCy model loaded")

    main(
        args.entity_csv,
        args.sent_csv,
        args.n_samples,
        nlp_,
        args.sep,
        args.shuffle,
        args.t_split,
    )

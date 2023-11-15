import pickle
from os import path
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sentence_transformers.losses import TripletDistanceMetric
import gc
import torch
import sys

sys.path.insert(0, path.abspath(".."))
from bi_encoder import BiEncoderTrainer, BiEncoderConfig

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-g", "--use-gpu", action="store_true")
    parser.add_argument("-a", "--use-amp", action="store_true")

    # Model name or path.
    parser.add_argument(
        "-m", "--base-model", type=str, default=BiEncoderConfig.BASE_MODEL
    )

    parser.add_argument(
        "-d",
        "--distance-metric",
        type=str,
        choices=["cosine", "euclidean", "manhattan"],
        default=BiEncoderConfig.DISTANCE_METRIC,
    )

    # Directory which contains train_examples.pkl and eval_examples.pkl. Each
    # file contains a list of sentence_transformers.InputExample objects,
    # for training and evaluation respectively.
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        # Server 115.
        default="/home/ec2-user/gc-experiments/2023_04_semantic_search/data/train_inputs",
    )

    # Directory to store output model and evaluation files.
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        # Server 115.
        default="/home/ec2-user/gc-experiments/2023_04_semantic_search/data/train_outputs",
    )

    # To only use a portion of the test and evaluation samples loaded.
    # If test-train-size is 0, uses all train and evaluation samples.
    parser.add_argument("-t", "--test-train-size", default=0, type=int)
    parser.add_argument("-u", "--test-eval-size", type=float, default=0.2)

    parser.add_argument("-l", "--checkpoint-limit", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-e", "--epochs", default=3, type=int)

    args = parser.parse_args()

    info(f"args: {args}")

    return args


def train(args):
    train_data, eval_data = load_inputs(args.input_dir)

    if args.test_train_size > 0:
        train_data, _ = train_test_split(
            train_data, train_size=args.test_train_size
        )
        _, eval_data = train_test_split(eval_data, test_size=0.2)

    distance_metric = transform_distance_arg(args.distance_metric)

    trainer = BiEncoderTrainer(
        base_model=args.base_model,
        use_gpu=args.use_gpu,
        distance_metric=distance_metric,
    )

    info("Starting training.")

    trainer.train(
        train_inputs=train_data,
        eval_inputs=eval_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
        checkpoint_limit=args.checkpoint_limit,
        use_amp=args.use_amp,
    )

    info("Finished training.")


def transform_distance_arg(arg: str) -> TripletDistanceMetric:
    if arg == "cosine":
        return TripletDistanceMetric.COSINE
    elif arg == "euclidean":
        return TripletDistanceMetric.EUCLIDEAN
    elif arg == "manhattan":
        return TripletDistanceMetric.MANHATTAN
    else:
        print(
            "Invalid distance metric arg. Must be one of 'cosine', 'euclidean', or 'manhattan'. Exiting."
        )
        sys.exit(1)


def load_inputs(dir_path):
    with open(path.join(dir_path, "train_examples.pkl"), "rb") as f:
        train_examples = pickle.load(f)

    with open(path.join(dir_path, "eval_examples.pkl"), "rb") as f:
        eval_examples = pickle.load(f)

    info(
        f"Loaded inputs.\nTrain: {len(train_examples)}\nEvaluation: {len(eval_examples)}"
    )

    return train_examples, eval_examples


def info(msg):
    print(f"\n\n----- train_bi_encoder.py - {msg}")


if __name__ == "__main__":
    args = parse_args()
    train(args)

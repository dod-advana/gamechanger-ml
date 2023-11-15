from argparse import ArgumentParser
from os.path import abspath
import pickle
from typing import List, Union
import sys

sys.path.insert(0, abspath(".."))
from bi_encoder import BiEncoderConfig, BiEncoderTrainingData
from query_classifier import QueryClassifier
from query_generator import QueryGenerator


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        default="/home/ec2-user/gc-experiments/2023_04_semantic_search/data/train_inputs/es_paragraphs.pkl",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="/home/ec2-user/gc-experiments/2023_04_semantic_search/data/train_inputs",
    )
    parser.add_argument("-e", "--eval-size", default=0.2, type=float)
    parser.add_argument(
        "-r", "--random-state", default=BiEncoderConfig.RANDOM_STATE, type=int
    )
    parser.add_argument(
        "-t",
        "--test",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--max-queries-per-passage",
        default=1,
        type=int,
    )

    args = parser.parse_args()

    info(f"args: {args}")

    return args


def make_training_data(
    passages: List[str],
    max_queries_per_passage: int,
    eval_size: Union[float, int],
    random_state: int,
    output_dir: str,
    input_size: int,
):
    info("Starting to make training data.")

    query_generator = QueryGenerator()
    query_classifier = QueryClassifier()
    data_creator = BiEncoderTrainingData(query_generator, query_classifier)

    if input_size > 0:
        passages = passages[:input_size]

    data_creator.create_examples(
        passages=passages,
        eval_size=eval_size,
        max_queries_per_passage=max_queries_per_passage,
        random_state=random_state,
        save_dir=output_dir,
    )

    info("Finished making training data.")


def load_pickle(pkl_path):
    info(f"Loading input from file: `{pkl_path}`")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return data


def info(msg):
    print(f"\n\n----- make_bi_encoder_training_data.py - {msg}")


if __name__ == "__main__":
    args = parse_args()
    paragraphs = load_pickle(args.input_path)
    make_training_data(
        passages=paragraphs,
        max_queries_per_passage=args.max_queries_per_passage,
        eval_size=args.eval_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
        input_size=args.test,
    )

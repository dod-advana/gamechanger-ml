from gamechangerml.configs import SemanticSearchConfig
from gamechangerml.src.search.semantic_search.train import (
    SemanticSearchFinetuner,
)
from gamechangerml.api.utils.pathselect import get_model_paths
import argparse
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

model_path_dict = get_model_paths()

LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
BASE_MODEL_NAME = SemanticSearchConfig.BASE_MODEL


def main(data_path, model_load_path, model_save_path, version):
    tuner = SemanticSearchFinetuner(
        model_load_path=model_load_path,
        model_save_path=model_save_path,
        data_directory=data_path,
        logger=logger,
        **SemanticSearchConfig.FINETUNE
    )
    return tuner.train(version)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Finetuning the sentence transformer model"
    )

    parser.add_argument(
        "--data-path",
        "-d",
        dest="data_path",
        required=True,
        help="path to csv with finetuning data",
    )

    parser.add_argument(
        "--model-load-path",
        "-m",
        dest="model_load_path",
        required=False,
        help="path to load model for fine-tuning",
    )

    parser.add_argument(
        "--model-save-path",
        "-s",
        dest="model_save_path",
        required=False,
        help="path to save model after fine-tuning",
    )

    parser.add_argument(
        "--version", "-v", dest="version", required=True, help="Model version"
    )

    args = parser.parse_args()

    ## getting default paths
    if args.model_load_path:
        model_load_path = args.model_load_path
    else:
        model_load_path = os.path.join(LOCAL_TRANSFORMERS_DIR, BASE_MODEL_NAME)

    if args.model_save_path:
        model_save_path = args.model_save_path
    else:
        model_save_path = model_load_path + str(
            datetime.now().strftime("%Y%m%d")
        )

    data_path = args.data_path

    logger.info(
        "\n|---------------------Beginning to finetune model-----------------------|"
    )

    main(data_path, model_load_path, model_save_path)

    logger.info(
        "|------------------------Done finetuning model--------------------------|\n"
    )

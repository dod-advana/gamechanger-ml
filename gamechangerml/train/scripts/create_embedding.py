from gamechangerml.src.search.sent_transformer.model import SentenceEncoder
from gamechangerml.src.utilities.arg_parser import LocalParser

from gamechangerml.src.utilities import utils as utils
from gamechangerml.src.utilities import aws_helper as aws_helper
from gamechangerml.api.utils.logger import logger

from datetime import datetime
from distutils.dir_util import copy_tree

import os
import torch
import json
from pathlib import Path
import tarfile
import typing as t
import subprocess

def create_tgz_from_dir(
    src_dir: t.Union[str, Path],
    dst_archive: t.Union[str, Path],
    exclude_junk: bool = False,
) -> None:
    with tarfile.open(dst_archive, "w:gz") as tar:
        tar.add(src_dir, arcname=os.path.basename(src_dir))

def create_embedding(corpus, existing_embeds = None, encoder_model = "msmarco-distilbert-base-v2", gpu = True, upload = False, version ="v4"):
    # Error fix for saving index and model to tgz
    # https://github.com/huggingface/transformers/issues/5486
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    except Exception as e:
        logger.warning(e)
    logger.info("Entered create embedding")

    # GPU check
    use_gpu = gpu
    if use_gpu and not torch.cuda.is_available:
        logger.info("GPU is not available. Setting `gpu` argument to False")
        use_gpu = False

    # Define model saving directories
    # here = os.path.dirname(os.path.realpath(__file__))
    # p = Path(here)
    model_dir = os.path.join("gamechangerml", "models")
    encoder_path = os.path.join(model_dir, "transformers", encoder_model)

    index_name = datetime.now().strftime("%Y%m%d")
    local_sent_index_dir = os.path.join(model_dir, "sent_index_" + index_name)

    # Define new index directory
    if not os.path.isdir(local_sent_index_dir):
        os.mkdir(local_sent_index_dir)

    # If existing index exists, copy content from reference index
    if existing_embeds is not None:
        copy_tree(existing_embeds, local_sent_index_dir)

    logger.info("Loading Encoder Model...")
    encoder = SentenceEncoder(encoder_path, use_gpu)
    logger.info("Creating Document Embeddings...")
    encoder.index_documents(corpus, local_sent_index_dir)

    try:
        user = os.environ.get("GC_USER", default="root")
        if (user =="root"):
            user = str(os.getlogin())
    except Exception as e:
        user = "unknown"
        logger.info("Could not get system user")
        logger.info(e)

    # Generating process metadata
    metadata = {
        "user": user,
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "doc_id_count": len(encoder.embedder.config["ids"]),
        "corpus_name": corpus,
        "encoder_model": encoder_model,
    }

    # Create metadata file
    metadata_path = os.path.join(local_sent_index_dir, "metadata.json")
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp)

    logger.info(f"Saved metadata.json to {metadata_path}")
    # Create .tgz file
    dst_path = local_sent_index_dir + ".tar.gz"
    create_tgz_from_dir(src_dir=local_sent_index_dir, dst_archive=dst_path)

    logger.info(f"Created tgz file and saved to {dst_path}")
    # Upload to S3
    if upload:
        # Loop through each file and upload to S3
        s3_sent_index_dir = f"gamechanger/models/sentence_index/{version}"
        logger.info(f"Uploading files to {s3_sent_index_dir}")
        logger.info(f"\tUploading: {local_sent_index_dir}")
        local_path = os.path.join(dst_path)
        s3_path = os.path.join(
            s3_sent_index_dir, "sent_index_" + index_name + ".tar.gz"
        )
        utils.upload_file(local_path, s3_path)
        logger.info(f"Successfully uploaded files to {s3_sent_index_dir}")

def main():
    parser = LocalParser()
    parser.add_argument(
        "-c",
        "--corpus",
        dest="corpus",
        required=True,
        type=str,
        help="Folder path containing GC Corpus",
    )
    parser.add_argument(
        "-e",
        "--existing-embeds",
        dest="existing_embeds",
        required=False,
        default=None,
        type=str,
        help="Folder path containing existing embeddings",
    )
    parser.add_argument(
        "-em",
        "--encoder-model",
        dest="encoder_model",
        required=False,
        default="msmarco-distilbert-base-v2",
        type=str,
        help="Encoder model used to encode the dataset",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest="gpu",
        required=False,
        default=True,
        type=bool,
        help="Boolean check if encoder model will be loaded to the GPU",
    )
    parser.add_argument(
        "-u",
        "--upload",
        dest="upload",
        required=False,
        default=False,
        type=bool,
        help="Boolean check if file will be uploaded to S3",
    )
    parser.add_argument(
        "-v",
        "--version",
        dest="version",
        required=False,
        default="v4",
        type=str,
        help="version string, must start with v, i.e. v1",
    )
    args = parser.parse_args()
    create_embedding(**args.__dict__)

    


if __name__ == "__main__":
    main()

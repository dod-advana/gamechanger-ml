import argparse
import logging
import os
from datetime import datetime, date

# import wikipedia
from gamechangerml.src.search.sent_transformer.model import SentenceEncoder
from gamechangerml.src.utilities.arg_parser import LocalParser

from gamechangerml.src.utilities import utils as utils
from gamechangerml.src.utilities import aws_helper as aws_helper
from gamechangerml.api.utils.logger import logger

from distutils.dir_util import copy_tree

import torch
import json
from pathlib import Path
import tarfile
import typing as t
import subprocess


from gamechangerml.src.search.query_expansion.build_ann_cli import (
    build_qe_model as bqe,
)
from gamechangerml.src.utilities import utils
from gamechangerml.configs.config import DefaultConfig, D2VConfig

# from gamechangerml.src.search.sent_transformer.model import SentenceEncoder
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

modelname = datetime.now().strftime("%Y%m%d")

SEARCH_MAPPINGS_FILE = "gamechangerml/data/SearchPdfMapping.csv"
TOPICS_FILE = "gamechangerml/data/topics_wiki.csv"
ORGS_FILE = "gamechangerml/data/agencies/agencies_in_corpus.csv"
data_path = "gamechangerml/data"
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception as e:
    logger.warning(e)
    logger.warning("MLFLOW may not be installed")


def lookup_wiki_summary(query):
    try:
        return wikipedia.summary(query).replace("\n", "")
    except:
        print(f"Could not retrieve description for {query}")
        return ""


class Pipeline:
    def __init__(self, steps={}):
        self.steps = steps
        POP_DOCS_PATH = Path(os.path.join(data_path, "popular_documents.csv"))
        if POP_DOCS_PATH.is_file():
            self.popular_docs = pd.read_csv(POP_DOCS_PATH)
        else:
            print("popular_documents.csv does not exist - generating meta data")
            self.create_metadata()
            self.popular_docs = pd.read_csv(POP_DOCS_PATH)

    def run_pipeline(
        self,
    ):
        for step in self.steps:
            logger.info("Running step %s in pipeline." % step)
            if step == "sentence":
                self.create_embedding(self.steps["sentence"])
            if step == "meta":
                self.create_metadata()
            if step == "qexp":
                self.create_qexp(self.steps["qexp"])

    def create_metadata(
        self,
    ):
        try:
            mappings = pd.read_csv(SEARCH_MAPPINGS_FILE)
        except Exception as e:
            print(e)
        mappings = self.process_mappings(mappings)
        mappings.to_csv(os.path.join(
            data_path, "popular_documents.csv"), index=False)
        try:
            topics = pd.read_csv(TOPICS_FILE)
            orgs = pd.read_csv(ORGS_FILE)
        except Exception as e:
            print(e)
        orgs.drop(columns=["Unnamed: 0"], inplace=True)
        topics.rename(
            columns={"name": "entity_name", "type": "entity_type"}, inplace=True
        )
        orgs.rename(columns={"Agency_Name": "entity_name"}, inplace=True)
        orgs["entity_type"] = "org"
        combined_ents = orgs.append(topics)
        combined_ents["information"] = combined_ents["entity_name"].apply(
            lambda x: lookup_wiki_summary(x)
        )
        combined_ents["information_source"] = "Wikipedia"
        combined_ents["information_retrieved"] = date.today().strftime(
            "%Y-%m-%d")
        combined_ents.to_csv(
            os.path.join(data_path, "combined_entities.csv"), index=False
        )

    def get_doc_count(self, filename: str):
        return self.popular_docs[self.popular_docs.doc == filename]["count"][0]

    def process_mappings(self, data):
        data = data.document.value_counts().to_frame().reset_index()
        data.rename(columns={"document": "count",
                    "index": "doc"}, inplace=True)
        return data

    def create_qexp(
        self,
        model_id,
        save_remote=False,
        corpus_dir=DefaultConfig.DATA_DIR,
        model_dest=DefaultConfig.LOCAL_MODEL_DIR,
        exp_name=modelname,
        validate=False,
        sentenceTrans=False,
        gpu=False,
    ):
        model_dir = model_dest

        # get model name schema
        model_id = utils.create_model_schema(model_dir, model_id)

        # todo: revise try/catch logic so mlflow_id is not referenced before assignment
        mlflow_id = None

        # start experiment
        try:
            # try to create experiment by exp name
            mlflow_id = mlflow.create_experiment(name=exp_name)
        except Exception as e:
            logger.warning(e)
            logger.warning("Cannot create experiment")
        # attempt mlflow start
        try:
            with mlflow.start_run(run_name=mlflow_id) as run:
                # build ANN indices
                index_dir = os.path.join(model_dest, model_id)
                bqe.main(
                    corpus_dir,
                    index_dir,
                    num_trees=125,
                    num_keywords=2,
                    ngram=(1, 2),
                    word_wt_file="word-freq-corpus-20201101.txt",
                    abbrv_file=None,
                )
                for param in D2VConfig.MODEL_ARGS:
                    mlflow.log_param(param, D2VConfig.MODEL_ARGS[param])
                mlflow.log_param("model_id", model_id)
                logger.info(
                    "-------------- Model Training Complete --------------")
                logger.info(
                    "-------------- Building Sentence Embeddings --------------"
                )
                if save_remote:
                    utils.save_all_s3(model_dest, model_id)

                if validate:
                    logger.info(
                        "-------------- Running Assessment Model Script --------------"
                    )

                    logger.info(
                        "-------------- Assessment is not available--------------"
                    )
                    """
                    results = mau.assess_model(
                        model_name=model_id,
                        logger=logger,
                        s3_corpus="corpus_20200909",
                        model_dir="gamechangerml/models/",
                        verbose=True,
                    )
                    for metric in results:
                        if metric != "model_name":
                            mlflow.log_metric(
                                key=metric, value=results[metric])
                    """
                    logger.info(
                        "-------------- Finished Assessment --------------")
                else:
                    logger.info(
                        "-------------- No Assessment Ran --------------")

            mlflow.end_run()
        except Exception:
            # try only models without mlflow
            logger.info(
                "-------------- Training without MLFLOW --------------")
            index_dir = os.path.join(model_dest, model_id)
            bqe.main(
                corpus_dir,
                index_dir,
                num_trees=125,
                num_keywords=2,
                ngram=(1, 2),
                word_wt_file="word-freq-corpus-20201101.txt",
                abbrv_file=None,
            )
            if save_remote:
                utils.save_all_s3(model_dest, model_id)
        logger.info("-------------- Model Training Complete --------------")

    def create_tgz_from_dir(
        self,
        src_dir: t.Union[str, Path],
        dst_archive: t.Union[str, Path],
        exclude_junk: bool = False,
    ) -> None:
        with tarfile.open(dst_archive, "w:gz") as tar:
            tar.add(src_dir, arcname=os.path.basename(src_dir))

    def create_embedding(
        self,
        corpus,
        existing_embeds=None,
        encoder_model="msmarco-distilbert-base-v2",
        gpu=True,
        upload=False,
        version="v4",
    ):
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
            logger.info(
                "GPU is not available. Setting `gpu` argument to False")
            use_gpu = False

        # Define model saving directories
        # here = os.path.dirname(os.path.realpath(__file__))
        # p = Path(here)
        model_dir = os.path.join("gamechangerml", "models")
        encoder_path = os.path.join(model_dir, "transformers", encoder_model)

        index_name = datetime.now().strftime("%Y%m%d")
        local_sent_index_dir = os.path.join(
            model_dir, "sent_index_" + index_name)

        # Define new index directory
        if not os.path.isdir(local_sent_index_dir):
            os.mkdir(local_sent_index_dir)
        logger.info(
            "-------------- Building Sentence Embeddings --------------")
        logger.info("Loading Encoder Model...")

        # If existing index exists, copy content from reference index
        if existing_embeds is not None:
            copy_tree(existing_embeds, local_sent_index_dir)
        try:
            mlflow.create_experiment("Sentence Embeddings")
        except Exception as e:
            logger.warning(e)
            logger.warning("Could not create experiment")
        try:
            with mlflow.start_run(run_name=index_name) as run:
                mlflow.log_param("model_id", index_name)
                encoder = SentenceEncoder(encoder_path, use_gpu)
                logger.info("Creating Document Embeddings...")
                encoder.index_documents(corpus, local_sent_index_dir)
                logger.info("-------------- Indexing Documents--------------")
                try:
                    user = os.environ.get("GC_USER", default="root")
                    if user == "root":
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
                metadata_path = os.path.join(
                    local_sent_index_dir, "metadata.json")
                with open(metadata_path, "w") as fp:
                    json.dump(metadata, fp)

                logger.info(f"Saved metadata.json to {metadata_path}")
                # Create .tgz file
                dst_path = local_sent_index_dir + ".tar.gz"
                self.create_tgz_from_dir(
                    src_dir=local_sent_index_dir, dst_archive=dst_path
                )

                logger.info(f"Created tgz file and saved to {dst_path}")
                for param in metadata:
                    mlflow.log_param(param, metadata[param])

            mlflow.end_run()
            logger.info(
                "-------------- Finished Sentence Embedding--------------")
        except Exception as e:
            print(e)
            logger.warning("Could not use MLFlow with this run")
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
            logger.info(
                "-------------- Finished Uploading Sentence Embedding--------------"
            )

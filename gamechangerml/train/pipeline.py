import argparse
import logging
import os
from datetime import datetime, date

# import wikipedia
from gamechangerml.src.search.sent_transformer.model import SentenceEncoder
from gamechangerml.src.search.query_expansion.qe import QE
from gamechangerml.src.utilities.arg_parser import LocalParser
from gamechangerml.src.model_testing.evaluation import IndomainRetrieverEvaluator, QexpEvaluator
from gamechangerml.scripts.finetune_sentence_retriever import STFinetuner

from gamechangerml.src.utilities import utils as utils
from gamechangerml.src.utilities import aws_helper as aws_helper
from gamechangerml.src.utilities.test_utils import get_user, get_most_recent_dir
from gamechangerml.api.utils.logger import logger
from gamechangerml.api.utils import processmanager
from gamechangerml.api.utils.pathselect import get_model_paths
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
from gamechangerml.configs.config import DefaultConfig, D2VConfig, QexpConfig, EmbedderConfig, SimilarityConfig

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
model_path_dict = get_model_paths()

SEARCH_MAPPINGS_FILE = "gamechangerml/data/SearchPdfMapping.csv"
TOPICS_FILE = "gamechangerml/data/topics_wiki.csv"
ORGS_FILE = "gamechangerml/data/agencies/agencies_in_corpus.csv"
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]

data_path = "gamechangerml/data"
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception as e:
    logger.warning(e)
    logger.warning("MLFLOW may not be installed")

try:
    import wikipedia
except Exception as e:
    logger.warning(e)
    logger.warning("Wikipedia may not be installed")


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
            logger.info(
                "popular_documents.csv does not exist - generating meta data")
            self.create_metadata()
            self.popular_docs = pd.read_csv(POP_DOCS_PATH)

        self.model_suffix = datetime.now().strftime("%Y%m%d")

    def run_pipeline(self, params):
        """
        run_pipeline: runs a list of configured components
        Args:
            params: dict of params
        Returns:
        """
        for step in self.steps:
            logger.info("Running step %s in pipeline." % step)
            if step == "sent_finetune":
                self.run(
                    build_type="sent_finetune", run_name=str(date.today()), params=params
                )
            if step == "sentence":
                self.run(
                    build_type="sentence", run_name=str(date.today()), params=params
                )
            if step == "meta":
                self.create_metadata()
            if step == "qexp":
                self.run(build_type="qexp", run_name=str(
                    date.today()), params=params)

    def create_metadata(
        self,
    ):
        """
        create_metadata: combines datasets to create a readable set for ingest
        Args:
        Returns:
        """
        try:
            mappings = pd.read_csv(SEARCH_MAPPINGS_FILE)
        except Exception as e:
            logger.info(e)
        mappings = self.process_mappings(mappings)
        mappings.to_csv(os.path.join(
            data_path, "popular_documents.csv"), index=False)
        try:
            topics = pd.read_csv(TOPICS_FILE)
            orgs = pd.read_csv(ORGS_FILE)
        except Exception as e:
            logger.info(e)
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
        data.rename(columns={"document": "pop_score",
                    "index": "doc"}, inplace=True)
        return data

    def finetune_sent(
        self,
        data_path=None,
        model=None,
        model_load_path=os.path.join(LOCAL_TRANSFORMERS_DIR, EmbedderConfig.MODEL_ARGS['encoder_model_name'])
    ):
        """
        finetune_sent: finetunes the sentence transformer - saves new model, a csv file of old/new cos sim scores,
        and a metadata file.
        Args:
            params and directories for finetuning the sentence transformer
        Returns:
            metadata: meta information on finetuning
        """
        model_save_path = model_load_path + '_' + str(date.today())
        if not data_path: # if no path to data, get most recent one
            data_parent = 'gamechangerml/data/training/sent_transformer'
            data_path = os.path.join(get_most_recent_dir(data_parent), 'training_data.json')
        finetuner = STFinetuner(
            model=model, model_load_path=model_load_path, model_save_path=model_save_path, **EmbedderConfig.FINETUNE
            )
        return finetuner.retrain(data_path)
    
    def create_qexp(
        self,
        model_id=None,
        upload=False,
        corpus=DefaultConfig.DATA_DIR,
        model_dest=DefaultConfig.LOCAL_MODEL_DIR,
        exp_name=modelname,
        validate=True,
        version="v4",
        gpu=False,
    ):
        """
        create_qexp: creates a query expansion model
        Args:
            params for qexp configuration
        Returns:
            metadata: params or meta information for qexp
            evals: evaluation results dict
        """
        model_dir = model_dest

        if not model_id:
            model_id = datetime.now().strftime("%Y%m%d")
        
        # get model name schema
        model_name = "qexp_" + model_id
        model_path = utils.create_model_schema(model_dir, model_name)
        evals = {"results": ""}
        params = D2VConfig.MODEL_ARGS
        try:
            # build ANN indices
            index_dir = os.path.join(model_dest, model_path)
            bqe.main(corpus, index_dir, **QexpConfig.MODEL_ARGS['bqe'])
            logger.info(
                "-------------- Model Training Complete --------------")
            # Create .tgz file
            dst_path = index_dir + ".tar.gz"
            self.create_tgz_from_dir(src_dir=index_dir, dst_archive=dst_path)

            logger.info(f"Created tgz file and saved to {dst_path}")

            if upload:
                S3_MODELS_PATH = "bronze/gamechanger/models"
                s3_path = os.path.join(S3_MODELS_PATH, f"qexp_model/{version}")
                self.upload(s3_path, dst_path, "qexp", model_id, version)

            if validate:
                logger.info(
                    "-------------- Running Assessment Model Script --------------"
                )
                #qxpeval = QexpEvaluator(qe_model_dir=index_dir, **QexpConfig.MODEL_ARGS['init'], **QexpConfig.MODEL_ARGS['expansion'], model=None)
                #evals = qxpeval.results
                
                logger.info(
                    "-------------- Assessment is not available--------------")
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
                logger.info("-------------- No Assessment Ran --------------")
        except Exception as e:
            logger.error(e)
            logger.error("Error with QExp building")
        return params, evals

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
        validate=True,
    ):
        """
        create_embedding: creates a sentence embedding
        Args:
            params for sentence configuration
        Returns:
            metadata: params or meta information for qexp
            evals: evaluation results dict
        """
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
        model_dir = os.path.join("gamechangerml", "models")
        encoder_path = os.path.join(model_dir, "transformers", encoder_model)

        model_id = datetime.now().strftime("%Y%m%d")
        model_name = "sent_index_" + model_id
        local_sent_index_dir = os.path.join(
            model_dir, model_name)

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
            overwrite = EmbedderConfig.MODEL_ARGS["overwrite"]
            min_token_len = EmbedderConfig.MODEL_ARGS["min_token_len"]
            return_id = EmbedderConfig.MODEL_ARGS["return_id"]
            verbose = EmbedderConfig.MODEL_ARGS["verbose"]
            encoder = SentenceEncoder(encoder_model_name=encoder_model, overwrite=overwrite, min_token_len=min_token_len, verbose=verbose, return_id=return_id, sent_index=local_sent_index_dir, use_gpu=use_gpu)
            logger.info("Creating Document Embeddings...")
            encoder.index_documents(corpus)
            logger.info("-------------- Indexing Documents--------------")
            user = get_user(logger)

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
            self.create_tgz_from_dir(
                src_dir=local_sent_index_dir, dst_archive=dst_path)

            logger.info(f"Created tgz file and saved to {dst_path}")
            logger.info(
                "-------------- Running Assessment Model Script --------------")

            sentev = IndomainRetrieverEvaluator(encoder=encoder, retriever=None, index=model_name, **EmbedderConfig.MODEL_ARGS, **SimilarityConfig.MODEL_ARGS)
            evals = sentev.results
            logger.info("evals: {}".format(str(evals)))
            
            logger.info(
                "-------------- Finished Sentence Embedding--------------")
        except Exception as e:
            logger.warning("Error with creating embedding")
            logger.error(e)
        # Upload to S3
        if upload:
            S3_MODELS_PATH = "bronze/gamechanger/models"
            s3_path = os.path.join(S3_MODELS_PATH, f"sentence_index/{version}")
            self.upload(s3_path, dst_path, "sentence_index", model_id, version)
        return metadata, evals

    def upload(self, s3_path, local_path, model_prefix, model_name, version):
        # Loop through each file and upload to S3
        logger.info(f"Uploading files to {s3_path}")
        logger.info(f"\tUploading: {local_path}")
        # local_path = os.path.join(dst_path)
        s3_path = os.path.join(
            s3_path, f"{model_prefix}_" + model_name + ".tar.gz")
        utils.upload_file(local_path, s3_path)
        logger.info(f"Successfully uploaded files to {s3_path}")
        logger.info("-------------- Finished Uploading --------------")

    def run(self, build_type, run_name, params):
        """
        run: record results of params and evaulations
        Args:
        Returns:
        """
        try:
            mlflow.create_experiment(str(date.today()))
        except Exception as e:
            logger.warning(e)
            logger.warning("Could not create experiment")
        try:
            with mlflow.start_run(run_name=run_name) as run:
                if build_type == "sent_finetune": 
                    metadata = self.finetune_sent(**params)
                elif build_type == "sentence":
                    metadata, evals = self.create_embedding(**params)
                elif build_type == "qexp":
                    metadata, evals = self.create_qexp(**params)
                self.mlflow_record(metadata, evals)
                processmanager.update_status(processmanager.training, 0, 1, "training" + build_type + " model")

            mlflow.end_run()
            processmanager.update_status(processmanager.training, 1, 1, "trained" + build_type + " model")
        except Exception as e:
            logger.warning(f"Error building {build_type} with MLFlow")
            logger.warning(e)
            logger.warning(f"Trying without MLFlow")
            try:
                if build_type == "sentence":
                    metadata, evals = self.create_embedding(**params)
                elif build_type == "qexp":
                    metadata, evals = self.create_qexp(**params)
            except Exception as err:
                logger.error("Could not train %s" % build_type)
                processmanager.update_status(
                    processmanager.loading_corpus, message="failed to load corpus", failed=True)
                processmanager.update_status(processmanager.training, message="failed to train " + build_type + " model", failed=True)

    def mlflow_record(self, metadata, evals):
        """
        mlflow_record: record results of params and evaulations
        Args:
        Returns:
        """
        for param in metadata:
            mlflow.log_param(param, metadata[param])
        for metric in evals:
            try:
                mlflow.log_metric(metric, evals[metric])
            except Exception as e:
                logger.warning(f"could not log metric: {metric}")

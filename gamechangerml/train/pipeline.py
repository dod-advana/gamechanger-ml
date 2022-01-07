import argparse
from gamechangerml import MODEL_PATH
from gamechangerml.api.fastapi.settings import CORPUS_DIR
from gamechangerml.src.search.ranking.ltr import LTR
from gamechangerml.src.featurization.topic_modeling import Topics
import logging
import os
from datetime import datetime, date
import time

# import wikipedia
from gamechangerml.src.search.sent_transformer.model import SentenceEncoder
from gamechangerml.src.search.query_expansion.qe import QE
from gamechangerml.src.utilities.arg_parser import LocalParser
from gamechangerml.src.model_testing.evaluation import (
    SQuADQAEvaluator,
    IndomainQAEvaluator,
    IndomainRetrieverEvaluator,
    MSMarcoRetrieverEvaluator,
    NLIEvaluator,
    QexpEvaluator,
)
from gamechangerml.scripts.finetune_sentence_retriever import STFinetuner

from gamechangerml.src.utilities import utils as utils
from gamechangerml.src.utilities import aws_helper as aws_helper
from gamechangerml.src.utilities.test_utils import (
    get_user,
    get_most_recent_dir,
    get_index_size,
    collect_evals,
    open_json,
)
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


from gamechangerml.src.search.query_expansion.build_ann_cli import build_qe_model as bqe
from gamechangerml.src.utilities import utils
from gamechangerml.configs.config import (
    DefaultConfig,
    D2VConfig,
    QexpConfig,
    QAConfig,
    EmbedderConfig,
    SimilarityConfig,
    QexpConfig,
)

import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s")
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

    # from mlflow.tracking import MlflowClient
except Exception as e:
    # logger.warning(e)
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
        logger.info(f"Could not retrieve description for {query}")
        return ""


class Pipeline:
    def __init__(self, steps={}):
        self.steps = steps
        POP_DOCS_PATH = Path(os.path.join(data_path, "popular_documents.csv"))
        if POP_DOCS_PATH.is_file():
            self.popular_docs = pd.read_csv(POP_DOCS_PATH)
        else:
            logger.info("popular_documents.csv does not exist - generating meta data")
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
                    build_type="sent_finetune",
                    run_name=str(date.today()),
                    params=params,
                )
            if step == "sentence":
                self.run(
                    build_type="sentence", run_name=str(date.today()), params=params
                )
            if step == "meta":
                self.create_metadata()
            if step == "qexp":
                self.run(build_type="qexp", run_name=str(date.today()), params=params)
            if step == "eval":
                self.run(build_type="eval", run_name=str(date.today()), params=params)
            if step == "topics":
                self.run(build_type="topics", run_name=str(date.today()), params=params)

    def create_metadata(self,):
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
        mappings.to_csv(os.path.join(data_path, "popular_documents.csv"), index=False)
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
        combined_ents["information_retrieved"] = date.today().strftime("%Y-%m-%d")
        combined_ents.to_csv(
            os.path.join(data_path, "combined_entities.csv"), index=False
        )

    def get_doc_count(self, filename: str):
        return self.popular_docs[self.popular_docs.doc == filename]["count"][0]

    def process_mappings(self, data):
        data = data.document.value_counts().to_frame().reset_index()
        data.rename(columns={"document": "pop_score", "index": "doc"}, inplace=True)
        return data

    def finetune_sent(
        self, batch_size=32, epochs=3, warmup_steps=100, testing_only=False
    ):
        """
        finetune_sent: finetunes the sentence transformer - saves new model, a csv file of old/new cos sim scores,
        and a metadata file.
        Args:
            params and directories for finetuning the sentence transformer
        Returns:
            metadata: meta information on finetuning
        """
        model_load_path = os.path.join(
            LOCAL_TRANSFORMERS_DIR, EmbedderConfig.BASE_MODEL
        )
        model_save_path = model_load_path + "_" + str(date.today())
        logger.info(f"Setting {str(model_save_path)} as save path for new model")
        data_path = get_most_recent_dir("gamechangerml/data/training/sent_transformer")
        logger.info(f"Loading in domain data to finetune from {data_path}")
        finetuner = STFinetuner(
            model_load_path=model_load_path,
            model_save_path=model_save_path,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            warmup_steps=warmup_steps,
        )
        logger.info("Loaded finetuner class...")
        logger.info(f"Testing only is set to: {testing_only}")
        return finetuner.retrain(data_path, testing_only)

    def evaluate(
        self, model_name, sample_limit, validation_data="latest", eval_type="original"
    ):
        """model_dict: {
        "model_name": [REQUIRED],
        "eval_type": ["original", "domain"],
        "sample_limit": 15000,
        "validation_data": "latest"
        }
        """

        def eval_qa(model_name, sample_limit, eval_type="original"):
            if eval_type == "original":
                logger.info(
                    f"Evaluating QA model on SQuAD dataset with sample limit of {str(sample_limit)}."
                )
                originalEval = SQuADQAEvaluator(
                    model_name=model_name,
                    sample_limit=sample_limit,
                    **QAConfig.MODEL_ARGS,
                )
                return originalEval.results
            elif eval_type == "domain":
                logger.info(
                    "No in-domain gamechanger evaluation available for the QA model."
                )
            else:
                logger.info(
                    "No eval_type selected. Options: ['original', 'gamechanger']."
                )

        def eval_sent(model_name, validation_data, eval_type="domain"):
            metadata = open_json(
                "metadata.json", os.path.join("gamechangerml/models", model_name)
            )
            encoder = metadata["encoder_model"]
            logger.info(f"Evaluating {model_name} created with {encoder}")
            if eval_type == "domain":
                if validation_data != "latest":
                    data_path = os.path.join(
                        "gamechangerml/data/validation/sent_transformer",
                        validation_data,
                    )
                else:
                    data_path = None
                results = {}
                for level in ["gold", "silver"]:
                    domainEval = IndomainRetrieverEvaluator(
                        index=model_name,
                        data_path=data_path,
                        data_level=level,
                        encoder_model_name=encoder,
                        sim_model_name=SimilarityConfig.BASE_MODEL,
                        **EmbedderConfig.MODEL_ARGS,
                    )
                    results[level] = domainEval.results
            elif eval_type == "original":
                originalEval = MSMarcoRetrieverEvaluator(
                    **EmbedderConfig.MODEL_ARGS,
                    encoder_model_name=EmbedderConfig.BASE_MODEL,
                    sim_model_name=SimilarityConfig.BASE_MODEL,
                )
                results = originalEval.results
            else:
                logger.info("No eval_type selected. Options: ['original', 'domain'].")

            return results

        def eval_sim(model_name, sample_limit, eval_type="original"):
            if eval_type == "original":
                logger.info(
                    f"Evaluating sim model on NLI dataset with sample limit of {str(sample_limit)}."
                )
                originalEval = NLIEvaluator(
                    sample_limit=sample_limit, sim_model_name=model_name
                )
                results = originalEval.results
                logger.info(f"Evals: {str(results)}")
                return results
            elif eval_type == "domain":
                logger.info("No in-domain evaluation available for the sim model.")
            else:
                logger.info("No eval_type selected. Options: ['original', 'domain'].")

        def eval_qe(model_name):
            domainEval = QexpEvaluator(
                qe_model_dir=os.path.join("gamechangerml/models", model_name),
                **QexpConfig.MODEL_ARGS["init"],
                **QexpConfig.MODEL_ARGS["expansion"],
            )
            results = domainEval.results
            logger.info(f"Evals: {str(results)}")
            return results

        results = {"original": {}, "domain": {}}
        try:
            logger.info(f"Attempting to evaluate model {model_name}")

            if "bert-base-cased-squad2" in model_name:
                results[eval_type] = eval_qa(model_name, sample_limit, eval_type)
            elif "msmarco-distilbert-base-v2" in model_name:
                results["original"] = eval_sent(
                    model_name, validation_data, eval_type="original"
                )
            elif "sent_index" in model_name:
                results["domain"] = eval_sent(
                    model_name, validation_data, eval_type="domain"
                )
            elif "distilbart-mnli-12-3" in model_name:
                results[eval_type] = eval_sim(model_name, sample_limit, eval_type)
            elif "qexp" in model_name:
                results["domain"] = eval_qe(model_name)
            else:
                logger.warning(
                    "There is currently no evaluation pipeline for this type of model."
                )
                raise Exception("No evaluation pipeline available")

        except Exception as e:
            logger.warning(f"Could not evaluate {model_name}")
            logger.warning(e)

        return results

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
            bqe.main(corpus, index_dir, **QexpConfig.MODEL_ARGS["bqe"])
            logger.info("-------------- Model Training Complete --------------")
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
                # qxpeval = QexpEvaluator(qe_model_dir=index_dir, **QexpConfig.MODEL_ARGS['init'], **QexpConfig.MODEL_ARGS['expansion'], model=None)
                # evals = qxpeval.results

                logger.info("-------------- Assessment is not available--------------")
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

                logger.info("-------------- Finished Assessment --------------")
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
        encoder_model="msmarco-distilbert-base-v2_2021-10-17",
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
            logger.info("GPU is not available. Setting `gpu` argument to False")
            use_gpu = False

        # Define model saving directories
        model_dir = os.path.join("gamechangerml", "models")
        encoder_path = os.path.join(model_dir, "transformers", encoder_model)

        model_id = datetime.now().strftime("%Y%m%d")
        model_name = "sent_index_" + model_id
        local_sent_index_dir = os.path.join(model_dir, model_name)

        # Define new index directory
        if not os.path.isdir(local_sent_index_dir):
            os.mkdir(local_sent_index_dir)
        logger.info("-------------- Building Sentence Embeddings --------------")
        logger.info("Loading Encoder Model...")

        # If existing index exists, copy content from reference index
        if existing_embeds is not None:
            copy_tree(existing_embeds, local_sent_index_dir)

        # Building the Index
        try:
            encoder = SentenceEncoder(
                encoder_model_name=encoder_model,
                use_gpu=use_gpu,
                transformer_path=LOCAL_TRANSFORMERS_DIR,
                **EmbedderConfig.MODEL_ARGS,
            )
            logger.info(
                f"Creating Document Embeddings with {encoder_model} on {corpus}"
            )
            logger.info("-------------- Indexing Documents--------------")
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            encoder.index_documents(corpus_path=corpus, index_path=local_sent_index_dir)
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("-------------- Completed Indexing --------------")
            user = get_user(logger)

            # Checking length of IDs
            try:
                SENT_INDEX_PATH = model_path_dict["sentence"]
                old_index_len = get_index_size(SENT_INDEX_PATH)
                new_index_len = len(encoder.embedder.config["ids"])
                if new_index_len < old_index_len:
                    logger.warning(
                        f"Length of index ({str(new_index_len)}) is shorter than previous index ({str(old_index_len)})"
                    )
                    logger.info(f"Old index location: {str(SENT_INDEX_PATH)}")
            except Exception as e:
                logger.warning(
                    f"Could not compare length to old index: {str(SENT_INDEX_PATH)}"
                )
                logger.error(e)

            # Generating process metadata
            metadata = {
                "user": user,
                "date_started": start_time,
                "date_finished": end_time,
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
            self.create_tgz_from_dir(src_dir=local_sent_index_dir, dst_archive=dst_path)

            logger.info(f"Created tgz file and saved to {dst_path}")
            logger.info("-------------- Running Evaluation --------------")

            try:
                evals = {}
                for level in ["gold", "silver"]:
                    sentev = IndomainRetrieverEvaluator(
                        encoder=encoder,
                        index=model_name,
                        data_level=level,
                        encoder_model_name=EmbedderConfig.BASE_MODEL,
                        sim_model_name=SimilarityConfig.BASE_MODEL,
                        **EmbedderConfig.MODEL_ARGS,
                    )
                    evals[level] = sentev.results
                    logger.info(
                        f"Evals for {level} standard validation: {(str(sentev.results))}"
                    )
            except Exception as e:
                logger.warning(
                    "Could not create evaluations for the new sentence index"
                )
                logger.error(e)

            logger.info("-------------- Finished Sentence Embedding--------------")
        except Exception as e:
            logger.warning("Error with creating embedding")
            logger.error(e)
        # Upload to S3
        if upload:
            S3_MODELS_PATH = "bronze/gamechanger/models"
            s3_path = os.path.join(S3_MODELS_PATH, f"sentence_index/{version}")
            self.upload(s3_path, dst_path, "sentence_index", model_id, version)
        return metadata, evals

    def init_ltr(self):
        try:
            ltr = LTR()
            logger.info("attempting to init LTR")
            resp = ltr.post_init_ltr()
            logger.info(resp)
            logger.info("attemtping to post features to ES")
            resp = ltr.post_features()
            logger.info(resp)
        except Exception as e:
            logger.warning(e)
            logger.warning("Could not initialize LTR")

    def create_ltr(self):
        try:
            ltr = LTR()
            processmanager.update_status(processmanager.ltr_creation, 0, 4)
            logger.info("Attempting to create judgement list")
            judgements = ltr.generate_judgement(ltr.mappings)
            processmanager.update_status(processmanager.ltr_creation, 1, 4)
            logger.info("Attempting to get features")
            fts = ltr.generate_ft_txt_file(judgements)
            processmanager.update_status(processmanager.ltr_creation, 2, 4)
            logger.info("Attempting to read in data")
            ltr.data = ltr.read_xg_data()
            logger.info("Attempting to train LTR model")
            bst, model = ltr.train()
            processmanager.update_status(processmanager.ltr_creation, 3, 4)
            logger.info("Created LTR model")
            with open(os.path.join(MODEL_PATH, "ltr/xgb-model.json")) as f:
                model = json.load(f)
            logger.info("removing old LTR")
            resp = ltr.delete_ltr("ltr_model")
            logger.info(resp)
            resp = ltr.post_model(model, model_name="ltr_model")
            logger.info("Posted LTR model")
            processmanager.update_status(processmanager.ltr_creation, 4, 4)
        except Exception as e:
            logger.error("Could not create LTR")

    def create_topics(self, sample_rate, upload=False):
        try:

            version = "v2"
            model_id = datetime.now().strftime("%Y%m%d")
            model_name = "topics_" + model_id
            model_dir = model_path_dict["topics"]

            local_dir = os.path.join(model_dir, model_name)
            # Define new index directory
            if not os.path.isdir(local_dir):
                os.mkdir(local_dir)

            # Train topics
            # TODO unwrap this like sentence encoder??
            topics_model = Topics()
            metadata = topics_model.train_from_files(
                corpus_dir=CORPUS_DIR, sample_rate=sample_rate, local_dir=local_dir
            )

            # Create metadata file
            metadata_path = os.path.join(local_dir, "metadata.json")
            with open(metadata_path, "w") as fp:
                json.dump(metadata, fp)

            logger.info(f"Saved metadata.json to {metadata_path}")
            # Create .tgz file
            tar_path = local_dir + ".tar.gz"
            self.create_tgz_from_dir(src_dir=local_dir, dst_archive=tar_path)

            evals = None  # TODO: figure out how to evaluate this
            logger.info("\n\n create_topics complete \n")

            # Upload to S3
            ## DAKOTA NOTE
            # # option on ml dash, is option b/c not needed running locally
            if upload:
                S3_MODELS_PATH = "bronze/gamechanger/models"
                s3_path = os.path.join(S3_MODELS_PATH, f"topics/{version}")
                self.upload(s3_path, tar_path, "topics", model_id, version)
            return metadata, evals

        except Exception as e:
            logger.error("Could not create topics", e)

    def upload(self, s3_path, local_path, model_prefix, model_name, version):
        # Loop through each file and upload to S3
        logger.info(f"Uploading files to {s3_path}")
        logger.info(f"\tUploading: {local_path}")
        # local_path = os.path.join(dst_path)
        s3_path = os.path.join(s3_path, f"{model_prefix}_" + model_name + ".tar.gz")
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
            # logger.warning(e)
            # logger.warning("Could not create experiment")
            pass
        try:
            with mlflow.start_run(run_name=run_name) as run:
                if build_type == "sent_finetune":
                    metadata, evals = self.finetune_sent(**params), {}
                elif build_type == "sentence":
                    metadata, evals = self.create_embedding(**params)
                elif build_type == "qexp":
                    metadata, evals = self.create_qexp(**params)
                elif build_type == "eval":
                    metadata, evals = {}, self.evaluate(**params)
                elif build_type == "topics":
                    metadata, evals = self.create_topics(**params)
                self.mlflow_record(metadata, evals)
                processmanager.update_status(
                    processmanager.training, 0, 1, "training" + build_type + " model"
                )

            mlflow.end_run()
            processmanager.update_status(
                processmanager.training, 1, 1, "trained" + build_type + " model"
            )
        except Exception as e:
            # logger.warning(f"Error building {build_type} with MLFlow")
            # logger.warning(e)
            logger.warning(f"Trying without MLFlow")
            try:
                if build_type == "sent_finetune":
                    metadata, evals = self.finetune_sent(**params), {}
                elif build_type == "sentence":
                    metadata, evals = self.create_embedding(**params)
                elif build_type == "qexp":
                    metadata, evals = self.create_qexp(**params)
                elif build_type == "eval":
                    metadata, evals = {}, self.evaluate(**params)
                elif build_type == "topics":
                    metadata, evals = self.create_topics(**params)
                else:
                    logger.info(
                        f"Started pipeline with unknown build_type: {build_type}"
                    )
                processmanager.update_status(
                    processmanager.training, 0, 1, "training" + build_type + " model"
                )
                processmanager.update_status(
                    processmanager.training, 1, 1, "trained" + build_type + " model"
                )
            except Exception as err:
                logger.error("Could not train %s" % build_type)
                logger.error(err)
                processmanager.update_status(
                    processmanager.loading_corpus,
                    message="failed to load corpus",
                    failed=True,
                )
                processmanager.update_status(
                    processmanager.training,
                    message="failed to train " + build_type + " model",
                    failed=True,
                )

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
                # logger.warning(f"could not log metric: {metric}")
                pass

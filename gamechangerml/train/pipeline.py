import logging
import os
import torch
import json
import urllib3
import pandas as pd
from distutils.dir_util import copy_tree
from datetime import datetime, date
from pathlib import Path
import typing as t

from gamechangerml.src.search.sent_transformer.model import SentenceEncoder
from gamechangerml.src.model_testing.evaluation import IndomainRetrieverEvaluator
from gamechangerml.scripts.finetune_sentence_retriever import STFinetuner
from gamechangerml.scripts.run_evaluation import eval_qa, eval_sent, eval_sim, eval_qe
from gamechangerml.src.featurization.make_meta import (
    make_pop_docs, make_combined_entities, make_corpus_meta
)
from gamechangerml.scripts.update_eval_data import make_tiered_eval_data
from gamechangerml.scripts.make_training_data import make_training_data

from gamechangerml.src.utilities import utils as utils
from gamechangerml.src.utilities import aws_helper as aws_helper
from gamechangerml.src.utilities.test_utils import get_user, get_most_recent_dir, get_index_size
from gamechangerml.api.utils.logger import logger
from gamechangerml.api.utils import processmanager
from gamechangerml.api.utils.pathselect import get_model_paths

from gamechangerml.src.search.query_expansion.build_ann_cli import (
    build_qe_model as bqe,
)
from gamechangerml.src.utilities import utils
from gamechangerml.configs.config import DefaultConfig, D2VConfig, QexpConfig, EmbedderConfig, SimilarityConfig, QexpConfig

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

LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
FEATURES_DATA_PATH = "gamechangerml/data/features"
USER_DATA_PATH = "gamechangerml/data/user_data"
PROD_DATA_FILE = "gamechangerml/data/features/generated_files/prod_test_data.csv"
SENT_INDEX = model_path_dict["sentence"]

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception as e:
    logger.warning(e)
    logger.warning("MLFLOW may not be installed")

class Pipeline:
    def __init__(self):

        self.model_suffix = datetime.now().strftime("%Y%m%d")
        ## read in input data files
        try:
            self.search_history = pd.read_csv(os.path.join(USER_DATA_PATH, "search_history/SearchPdfMapping.csv"))
            self.topics = pd.read_csv(os.path.join(FEATURES_DATA_PATH, "topics_wiki.csv"))
            self.orgs = pd.read_csv(os.path.join(FEATURES_DATA_PATH, "agencies.csv"))
        except Exception as e:
            logger.info(e)
        
        ## set paths for output data files
        self.pop_docs_path = Path(os.path.join(FEATURES_DATA_PATH, "popular_documents.csv"))
        self.combined_ents_path = Path(os.path.join(FEATURES_DATA_PATH, "combined_entities.csv"))

    def run_pipeline(self, steps={}):
        """
        run_pipeline: runs a list of configured components
        Args:
            steps: Dictionary of steps (build_types) and their function args/params, ex:
            {
                "meta": {"corpus_dir": "gamechangerml/corpus", "meta_steps": ["rank_features"]},
                "sent_finetune": {...}
            }
        Returns:
        """
        step_order = ["meta", "sent_finetune", "qexp", "sentence", "eval"]
        steps = {k: steps[k] for k in step_order if k in steps}
        for step in steps:
            try:
                logger.info("Running step %s in pipeline." % step)
                self.run(build_type=step, run_name=str(date.today()), params=steps[step])
            except Exception as e:
                logger.info(f"Could not run {step}")
                logger.info(e)
    
    def create_metadata(
        self,
        meta_steps,
        corpus_dir:str='gamechangerml/corpus',
        index_path:str='gamechangerml/models/sent_index_20210715',
        days: int=80,
        prod_data_file=PROD_DATA_FILE,
        n_returns: int=15,
        n_matching: int=3,
        level: str='silver',
        update_eval_data: bool=False,
        retriever=None,
        upload=True
    ) -> None:
        """
        create_metadata: combines datasets to create readable sets for ingest
        Args: 
            corpus_dir [Union[str,os.PathLike]]: path to corpus JSONs
            meta_steps [List[str]]: list of metadata steps to execute (
                options: ["pop_docs", "combined_ents", "rank_features", "update_sent_data"])
            days [int]: days back to go for creating rank features (** rank_features)
            prod_data_file [Union[str,os.PathLike]]: path to prod data file (** rank_features)
            index_path [Union[str,os.PathLike]]: sent index path (** update_sent_data)
            n_returns [int]: number of neutral (non-matching) paragraphs to retrieve (** update_sent_data)
            n_matching [int]: number of matching paragraphs to retrieve (** update_sent_data)
            level [str]: level of tiered eval data to use (any, silver, gold) (** update_sent_data)
            update_eval_data [bool]: whether or not to update the eval data (** update_sent_data)
        Returns:
            None (saves files for each step)
        """
        logger.info(f"Meta steps: {str(meta_steps)}")

        if "pop_docs" in meta_steps:
            make_pop_docs(self.search_history, self.pop_docs_path)
        if "combined_ents" in meta_steps:
            make_combined_entities(self.topics, self.orgs, self.combined_ents_path)
        if "rank_features" in meta_steps:
            make_corpus_meta(corpus_dir, days, prod_data_file)
        if "update_sent_data" in meta_steps:
            make_training_data(index_path, n_returns, n_matching, level, update_eval_data, retriever)

    def finetune_sent(
        self,
        batch_size: int=32,
        epochs: int=3,
        warmup_steps: int=100,
        testing_only: bool=False
    ) -> t.Dict[str,str]:
        """finetune_sent: finetunes the sentence transformer - saves new model, 
           a csv file of old/new cos sim scores, and a metadata file.
        Args:
            batch_size [int]: batch_size
            epochs [int]: epochs
            warmup_steps [int]: warmup steps
            testing_only [bool]: set True if only testing finetune functionality
        Returns:
            metadata: meta information on finetuning
        """
        model_load_path=os.path.join(LOCAL_TRANSFORMERS_DIR, EmbedderConfig.BASE_MODEL)
        model_save_path = model_load_path + "_" + str(date.today())
        logger.info(f"Setting {str(model_save_path)} as save path for new model")
        data_path = get_most_recent_dir("gamechangerml/data/training/sent_transformer")
        logger.info(f"Loading in domain data to finetune from {data_path}")
        finetuner = STFinetuner(
            model_load_path=model_load_path, model_save_path=model_save_path, shuffle=True, batch_size=batch_size, epochs=epochs, warmup_steps=warmup_steps
            )
        logger.info("Loaded finetuner class...")
        logger.info(f"Testing only is set to: {testing_only}")
        return finetuner.retrain(data_path, testing_only)

    def evaluate(
        self,
        model_name: str,
        sample_limit: int,
        validation_data: str="latest",
        eval_type: str="original"
    ) -> t.Dict[str,str]:
        """Evaluates models/sent index
        Args:
            model_name [str]: name of the model or sent index to evaluate
            sample_limit [int]: max samples for evaluating (if testing on original data)
            validation_data [str]: which validation set to use (
                "latest" pulls newest, otherwise needs timestamp dir ex. '2021-12-01_191740')
            eval_type [str]: type of evaluation to run (options = ["original", "domain"])
        Returns:
            eval [Dict[str,str]]: evaluation dictionary
        """
        results = {"original": {}, "domain": {}}
        try:
            logger.info(f"Attempting to evaluate model {model_name}")
            
            if "bert-base-cased-squad2" in model_name:
                results[eval_type] = eval_qa(model_name, sample_limit, eval_type)
            elif "msmarco-distilbert-base-v2" in model_name:
                results["original"] = eval_sent(model_name, validation_data, eval_type="original")
            elif "sent_index" in model_name:
                results["domain"] = eval_sent(model_name, validation_data, eval_type="domain")
            elif "distilbart-mnli-12-3" in model_name:
                results[eval_type] = eval_sim(model_name, sample_limit, eval_type)
            elif 'qexp' in model_name:
                results['domain'] = eval_qe(model_name)
            else:
                logger.warning("There is currently no evaluation pipeline for this type of model.")
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
            logger.info(
                "-------------- Model Training Complete --------------")
            # Create .tgz file
            dst_path = index_dir + ".tar.gz"
            utils.create_tgz_from_dir(src_dir=index_dir, dst_archive=dst_path)

            logger.info(f"Created tgz file and saved to {dst_path}")

            if upload:
                S3_MODELS_PATH = "bronze/gamechanger/models"
                s3_path = os.path.join(S3_MODELS_PATH, f"qexp_model/{version}")
                utils.upload(s3_path, dst_path, "qexp", model_id, version)

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

    def create_embedding(
        self,
        corpus,
        existing_embeds=None,
        encoder_model="msmarco-distilbert-base-v2_2021-10-17",
        gpu=True,
        upload=False,
        version="v4",
        validate=True
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

        # Building the Index
        try:
            encoder = SentenceEncoder(encoder_model_name=encoder_model, use_gpu=use_gpu, transformer_path=LOCAL_TRANSFORMERS_DIR, **EmbedderConfig.MODEL_ARGS)
            logger.info(f"Creating Document Embeddings with {encoder_model} on {corpus}")
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
                    logger.warning(f"Length of index ({str(new_index_len)}) is shorter than previous index ({str(old_index_len)})")
                    logger.info(f"Old index location: {str(SENT_INDEX_PATH)}")
            except Exception as e:
                logger.warning(f"Could not compare length to old index: {str(SENT_INDEX_PATH)}")
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
            utils.create_tgz_from_dir(
                src_dir=local_sent_index_dir, dst_archive=dst_path)

            logger.info(f"Created tgz file and saved to {dst_path}")
            logger.info("-------------- Running Evaluation --------------")

            try:
                evals = {}
                for level in ['gold', 'silver']:
                    sentev = IndomainRetrieverEvaluator(encoder=encoder, index=model_name, data_level=level, encoder_model_name=EmbedderConfig.BASE_MODEL, sim_model_name=SimilarityConfig.BASE_MODEL, **EmbedderConfig.MODEL_ARGS)
                    evals[level] = sentev.results
                    logger.info(f"Evals for {level} standard validation: {(str(sentev.results))}")
            except Exception as e:
                logger.warning("Could not create evaluations for the new sentence index")
                logger.error(e)
            
            logger.info(
                "-------------- Finished Sentence Embedding--------------")
        except Exception as e:
            logger.warning("Error with creating embedding")
            logger.error(e)
        # Upload to S3
        if upload:
            S3_MODELS_PATH = "bronze/gamechanger/models"
            s3_path = os.path.join(S3_MODELS_PATH, f"sentence_index/{version}")
            utils.upload(s3_path, dst_path, "sentence_index", model_id, version)
        return metadata, evals

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
        metadata = evals = {}
        try:
            with mlflow.start_run(run_name=run_name) as run:
                if build_type == "sent_finetune": 
                    metadata = self.finetune_sent(**params)
                elif build_type == "sentence":
                    metadata, evals = self.create_embedding(**params)
                elif build_type == "qexp":
                    metadata, evals = self.create_qexp(**params)
                elif build_type == "eval":
                    evals = self.evaluate(**params)
                elif build_type == "meta":
                    self.create_metadata(**params)
                self.mlflow_record(metadata, evals)
                processmanager.update_status(processmanager.training, 0, 1, "training" + build_type + " model")
            mlflow.end_run()
            processmanager.update_status(processmanager.training, 1, 1, "trained" + build_type + " model")
        except Exception as e:
            logger.warning(f"Error building {build_type} with MLFlow")
            logger.warning(e)
            logger.warning(f"Trying without MLFlow")
            try:
                if build_type == "sent_finetune": 
                    metadata = self.finetune_sent(**params)
                elif build_type == "sentence":
                    metadata, evals = self.create_embedding(**params)
                elif build_type == "qexp":
                    metadata, evals = self.create_qexp(**params)
                elif build_type == "eval":
                    evals = self.evaluate(**params)
                elif build_type == "meta":
                    self.create_metadata(**params)
                else:
                    logger.info(f"Started pipeline with unknown build_type: {build_type}")
                processmanager.update_status(processmanager.training, 0, 1, "training" + build_type + " model")
                processmanager.update_status(processmanager.training, 1, 1, "trained" + build_type + " model")
            except Exception as err:
                logger.error("Could not train %s" % build_type)
                processmanager.update_status(
                    processmanager.loading_corpus, message="failed to load corpus", failed=True)
                processmanager.update_status(
                    processmanager.training, message="failed to train " + build_type + " model", failed=True)

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

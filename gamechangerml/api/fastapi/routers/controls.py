from fastapi import APIRouter, Response, status
import subprocess
import os
import json
from datetime import datetime, date
from gamechangerml.src.utilities import utils
from gamechangerml.api.fastapi.model_config import Config
from gamechangerml.api.fastapi.version import __version__
from gamechangerml.api.fastapi.settings import *
from gamechangerml.api.fastapi.routers.startup import *
from gamechangerml.api.utils.threaddriver import MlThread
from gamechangerml.train.pipeline import Pipeline
from gamechangerml.src.search.ranking.ltr import LTR
from gamechangerml.api.utils import processmanager
from gamechangerml.api.fastapi.model_loader import ModelLoader
from gamechangerml.src.utilities.test_utils import (
    collect_evals,
    open_json,
    get_most_recent_dir,
    collect_sent_evals_gc,
    handle_sent_evals,
)

from gamechangerml.src.search.sent_transformer.finetune import STFinetuner
from gamechangerml.src.model_testing.evaluation import (
    SQuADQAEvaluator,
    IndomainQAEvaluator,
    IndomainRetrieverEvaluator,
    MSMarcoRetrieverEvaluator,
    NLIEvaluator,
    QexpEvaluator,
)
from gamechangerml.configs.config import (
    QAConfig,
    EmbedderConfig,
    SimilarityConfig,
    QexpConfig,
)

router = APIRouter()
MODELS = ModelLoader()
ltr = MODELS.ltr_model
## Get Methods ##


@router.get("/")
async def api_information():
    return {
        "API": "FOR TRANSFORMERS",
        "API_Name": "GAMECHANGER ML API",
        "Version": __version__,
    }


@router.get("/getProcessStatus")
async def get_process_status():
    return {
        "process_status": processmanager.PROCESS_STATUS.value,
        "completed_process": processmanager.COMPLETED_PROCESS.value,
    }


@router.get("/getModelsList")
def get_downloaded_models_list():
    qexp_list = {}
    sent_index_list = {}
    transformer_list = {}
    try:
        for f in os.listdir(Config.LOCAL_PACKAGED_MODELS_DIR):
            if ("qexp_" in f) and ("tar" not in f):
                qexp_list[f] = {}
                meta_path = os.path.join(
                    Config.LOCAL_PACKAGED_MODELS_DIR, f, "metadata.json"
                )
                if os.path.isfile(meta_path):
                    meta_file = open(meta_path)
                    qexp_list[f] = json.load(meta_file)
                    qexp_list[f]["evaluation"] = {}
                    qexp_list[f]["evaluation"] = collect_evals(
                        os.path.join(Config.LOCAL_PACKAGED_MODELS_DIR, f)
                    )
                    meta_file.close()
    except Exception as e:
        logger.error(e)
        logger.info("Cannot get QEXP model path")

    # TRANSFORMER MODEL PATH
    try:
        for trans in os.listdir(LOCAL_TRANSFORMERS_DIR.value):
            if trans not in ignore_files and "." not in trans:
                transformer_list[trans] = {}
                config_path = os.path.join(
                    LOCAL_TRANSFORMERS_DIR.value, trans, "config.json"
                )
                if os.path.isfile(config_path):
                    config_file = open(config_path)
                    transformer_list[trans] = json.load(config_file)
                    transformer_list[trans]["evaluation"] = {}
                    transformer_list[trans]["evaluation"] = collect_evals(
                        os.path.join(LOCAL_TRANSFORMERS_DIR.value, trans)
                    )
                    config_file.close()
    except Exception as e:
        logger.error(e)
        logger.info("Cannot get TRANSFORMER model path")
    # SENTENCE INDEX
    # get largest file name with sent_index prefix (by date)
    try:
        for f in os.listdir(Config.LOCAL_PACKAGED_MODELS_DIR):
            if ("sent_index" in f) and ("tar" not in f):
                logger.info(f"sent indices: {str(f)}")
                sent_index_list[f] = {}
                meta_path = os.path.join(
                    Config.LOCAL_PACKAGED_MODELS_DIR, f, "metadata.json"
                )
                if os.path.isfile(meta_path):
                    meta_file = open(meta_path)
                    sent_index_list[f] = json.load(meta_file)
                    sent_index_list[f]["evaluation"] = {}

                    sent_index_list[f]["evaluation"] = handle_sent_evals(
                        os.path.join(Config.LOCAL_PACKAGED_MODELS_DIR, f)
                    )
                    meta_file.close()
    except Exception as e:
        logger.error(e)
        logger.info("Cannot get Sentence Index model path")
    model_list = {
        "transformers": transformer_list,
        "sentence": sent_index_list,
        "qexp": qexp_list,
    }
    return model_list


@router.get("/LTR/initLTR", status_code=200)
async def initLTR(response: Response):
    """generate judgement - checks how many files are in the corpus directory
    Args:
    Returns: integer
    """
    number_files = 0
    resp = None
    try:

        logger.info("Attempting to initialize LTR")
        resp = ltr.post_init_ltr()
        logger.info(resp)
        logger.info("Attempting to post features to LTR")
        resp = ltr.post_features()
        logger.info(resp)
    except Exception as e:
        logger.warning("Could not init LTR")


@router.get("/LTR/createModel", status_code=200)
async def create_LTR_model(response: Response):
    """generate judgement - checks how many files are in the corpus directory
    Args:
    Returns: integer
    """
    number_files = 0
    resp = None
    try:
        model = []

        def ltr_process():
            logger.info("Attempting to create judgement list")
            judgements = ltr.generate_judgement(ltr.mappings)
            logger.info("Attempting to get features")

            fts = ltr.generate_ft_txt_file(judgements)
            logger.info("Attempting to read in data")
            ltr.data = ltr.read_xg_data()
            logger.info("Attempting to train LTR model")
            bst, model = ltr.train()
            logger.info("Created LTR model")
            with open("gamechangerml/models/ltr/xgb-model.json") as f:
                model = json.load(f)
            logger.info("removing old LTR")
            resp = ltr.delete_ltr("ltr_model")
            logger.info(resp)
            resp = ltr.post_model(model, model_name="ltr_model")
            logger.info("Posted LTR model")

        ltr_thread = MlThread(ltr_process)

        ltr_thread.start()

    except Exception as e:
        logger.warning(e)
        logger.warning(f"There is an issue with LTR creation")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp


@router.get("/getFilesInCorpus", status_code=200)
async def files_in_corpus(response: Response):
    """files_in_corpus - checks how many files are in the corpus directory
    Args:
    Returns: integer
    """
    number_files = 0
    try:
        logger.info("Attempting to download dependencies from S3")
        number_files = len(
            [
                name
                for name in os.listdir(CORPUS_DIR)
                if os.path.isfile(os.path.join(CORPUS_DIR, name))
            ]
        )
    except:
        logger.warning(f"Could not get dependencies from S3")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return json.dumps(number_files)


@router.get("/getCurrentTransformer")
async def get_trans_model():
    """get_trans_model - endpoint for current transformer
    Args:
    Returns:
        dict of model name
    """
    # sent_model = latest_intel_model_sent.value
    return {
        "sim_model": latest_intel_model_sim.value,
        "encoder_model": latest_intel_model_encoder.value,
        "sentence_index": SENT_INDEX_PATH.value,
        "qexp_model": QEXP_MODEL_NAME.value,
        "qa_model": latest_qa_model.value,
    }


@router.get("/download", status_code=200)
async def download(response: Response):
    """download - downloads dependencies from s3
    Args:
    Returns:
    """
    try:
        logger.info("Attempting to download dependencies from S3")
        output = subprocess.call(
            ["gamechangerml/scripts/download_dependencies.sh"])
        # get_transformers(overwrite=False)
        # get_sentence_index(overwrite=False)
    except:
        logger.warning(f"Could not get dependencies from S3")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return


@router.get("/s3", status_code=200)
async def s3_func(function, response: Response):
    """s3_func - s3 functionality for model managment
    Args:
        function: str
    Returns:
    """
    models = []
    try:
        logger.info("Attempting to download dependencies from S3")
        s3_path = "bronze/gamechanger/models/"
        if function == "models":
            models = utils.get_models_list(s3_path)
    except:
        logger.warning(f"Could not get dependencies from S3")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return models


## Post Methods ##


@router.post("/reloadModels", status_code=200)
async def reload_models(model_dict: dict, response: Response):
    """load_latest_models - endpoint for updating the transformer model
    Args:
        model_dict: dict; {"sentence": "bert...", "qexp": "bert...", "transformer": "bert..."}

        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
    """
    try:
        total = len(model_dict)
        processmanager.update_status(processmanager.reloading, 0, total)
        # put the reload process on a thread

        def reload_thread(model_dict):
            try:
                progress = 0
                if "sentence" in model_dict:
                    sentence_path = os.path.join(
                        Config.LOCAL_PACKAGED_MODELS_DIR, model_dict["sentence"]
                    )
                    # uses SENT_INDEX_PATH by default
                    logger.info("Attempting to load Sentence Transformer")
                    MODELS.initSentenceSearcher(sentence_path)
                    SENT_INDEX_PATH.value = sentence_path
                    progress += 1
                    processmanager.update_status(
                        processmanager.reloading, progress, total
                    )
                if "qexp" in model_dict:
                    qexp_name = os.path.join(
                        Config.LOCAL_PACKAGED_MODELS_DIR, model_dict["qexp"]
                    )
                    # uses QEXP_MODEL_NAME by default
                    logger.info("Attempting to load QE")
                    MODELS.initQE(qexp_name)
                    QEXP_MODEL_NAME.value = qexp_name
                    progress += 1
                    processmanager.update_status(
                        processmanager.reloading, progress, total
                    )
            except Exception as e:
                logger.warning(e)
                processmanager.update_status(
                    processmanager.reloading, failed=True)

        args = {"model_dict": model_dict}
        thread = MlThread(reload_thread, args)
        thread.start()
    except Exception as e:
        logger.warning(e)

    return await get_process_status()


@router.post("/downloadCorpus", status_code=200)
async def download_corpus(corpus_dict: dict, response: Response):
    """load_latest_models - endpoint for updating the transformer model
    Args:
        model_dict: dict; {"sentence": "bert...", "qexp": "bert...", "transformer": "bert..."}

        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
    """
    try:
        logger.info("Attempting to download corpus from S3")
        # grabs the s3 path to the corpus from the post in "corpus"
        # then passes in where to dowload the corpus locally.
        args = {"corpus_dir": corpus_dict["corpus"], "output_dir": CORPUS_DIR}
        processmanager.update_status(processmanager.corpus_download)
        corpus_thread = MlThread(utils.get_s3_corpus, args)
        corpus_thread.start()
    except:
        logger.warning(f"Could not get corpus from S3")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return await get_process_status()


@router.post("/trainModel", status_code=200)
async def train_model(model_dict: dict, response: Response):
    """load_latest_models - endpoint for updating the transformer model
    Args:
        model_dict: dict; {"encoder_model":"msmarco-distilbert-base-v2", "gpu":true, "upload":false,"version": "v5"}

        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
    """
    try:
        # Methods for all the different models we can train
        def update_metadata(model_dict=model_dict):
            logger.info("Attempting to update feature metadata")
            pipeline = Pipeline()
            model_dict["build_type"] = "meta"
            try:
                corpus_dir = model_dict["corpus_dir"]
            except:
                corpus_dir = CORPUS_DIR
            try:
                meta_steps = model_dict["meta_steps"]
            except:
                meta_steps = ["pop_docs", "combined_ents", "rank_features"]
            args = {
                "meta_steps": meta_steps,
                "corpus_dir": corpus_dir
            }
            pipeline.run(
                build_type=model_dict["build_type"], 
                run_name=datetime.now().strftime("%Y%m%d"), 
                params=args)

        def finetune_sentence(model_dict=model_dict):
            logger.info("Attempting to finetune the sentence transformer")
            pipeline = Pipeline()
            try:
                testing_only = model_dict["testing_only"]
            except:
                testing_only = False
            args = {
                "batch_size": model_dict["batch_size"],
                "epochs": model_dict["epochs"],
                "warmup_steps": model_dict["warmup_steps"],
                "testing_only": testing_only,
            }
            pipeline.run(
                build_type="sent_finetune",
                run_name=datetime.now().strftime("%Y%m%d"),
                params=args,
            )

        def train_sentence(model_dict=model_dict):
            logger.info("Attempting to start sentence pipeline")
            pipeline = Pipeline()
            try:
                corpus_dir = model_dict["corpus_dir"]
            except:
                corpus_dir = CORPUS_DIR
            if not os.path.exists(corpus_dir):
                logger.warning(
                    f"Corpus is not in local directory {str(corpus_dir)}")
                raise Exception("Corpus is not in local directory")
            args = {
                "corpus": corpus_dir,
                "encoder_model": model_dict["encoder_model"],
                "gpu": bool(model_dict["gpu"]),
                "upload": bool(model_dict["upload"]),
                "version": model_dict["version"],
            }
            pipeline.run(
                build_type=model_dict["build_type"],
                run_name=datetime.now().strftime("%Y%m%d"),
                params=args,
            )

        def train_qexp(model_dict=model_dict):
            logger.info("Attempting to start qexp pipeline")
            pipeline = Pipeline()
            args = {
                "model_id": model_dict["model_id"],
                "validate": bool(model_dict["validate"]),
                "upload": bool(model_dict["upload"]),
                "version": model_dict["version"],
            }
            pipeline.run(
                build_type=model_dict["build_type"],
                run_name=datetime.now().strftime("%Y%m%d"),
                params=args,
            )

        def run_evals(model_dict=model_dict):
            logger.info("Attempting to run evaluation")
            pipeline = Pipeline()
            args = {
                "model_name": model_dict["model_name"],
                "eval_type": model_dict["eval_type"],
                "sample_limit": model_dict["sample_limit"],
                "validation_data": model_dict["validation_data"],
            }
            pipeline.run(
                build_type=model_dict["build_type"],
                run_name=datetime.now().strftime("%Y%m%d"),
                params=args,
            )

        # Create a mapping between the training methods and input from the api
        training_switch = {
            "sentence": train_sentence,
            "qexp": train_qexp,
            "sent_finetune": finetune_sentence,
            "eval": run_evals,
            "meta": update_metadata
        }
        # Set the training method to be loaded onto the thread
        if "build_type" in model_dict and model_dict["build_type"] in training_switch:
            training_method = training_switch[model_dict["build_type"]]
        else:  # PLACEHOLDER
            model_dict["build_type"] = "sentence"
            training_method = training_switch[model_dict["build_type"]]

        training_thread = MlThread(training_method)
        training_thread.start()

    except:
        logger.warning(f"Could not train/evaluate the model")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return await get_process_status()

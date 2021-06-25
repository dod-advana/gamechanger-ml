from fastapi import APIRouter, Response, status
import subprocess
import os
from gamechangerml.src.utilities import utils
from gamechangerml.api.fastapi.model_config import Config
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.api.fastapi.version import __version__
from gamechangerml.api.fastapi.settings import *
from gamechangerml.api.fastapi.routers.startup import *

router = APIRouter()

## Get Methods ##

@router.get("/")
async def api_information():
    return {
        "API": "FOR TRANSFORMERS",
        "API_Name": "GAMECHANGER ML API",
        "Version": __version__
    }
@router.get("/getModelsList")
def get_downloaded_models_list():
    qexp_list = []
    sent_index_list = []
    transformer_list = []
    try:
        qexp_list = [
            f
            for f in os.listdir(Config.LOCAL_PACKAGED_MODELS_DIR)
            if ("qexp_" in f) and ("tar" not in f)
        ]
        qexp_list.sort(reverse=True)
    except Exception as e:
        logger.error(e)
        logger.info("Cannot get QEXP model path")

    # TRANSFORMER MODEL PATH
    try:
        transformer_list = [
            trans
            for trans in os.listdir(LOCAL_TRANSFORMERS_DIR)
            if trans not in ignore_files and '.' not in trans
        ]
    except Exception as e:
        logger.error(e)

        logger.info("Cannot get TRANSFORMER model path")
    # SENTENCE INDEX
    # get largest file name with sent_index prefix (by date)
    try:
        sent_index_list = [
            f
            for f in os.listdir(Config.LOCAL_PACKAGED_MODELS_DIR)
            if ("sent_index" in f) and ("tar" not in f)
        ]
        sent_index_list.sort(reverse=True)
    except Exception as e:
        logger.error(e)
        logger.info("Cannot get Sentence Index model path")
    model_list = {
        "transformers": transformer_list,
        "sentence": sent_index_list,
        "qexp": qexp_list,
    }
    return model_list


@router.get("/getCurrentTransformer")
async def get_trans_model():
    """get_trans_model - endpoint for current transformer
    Args:
    Returns:
        dict of model name
    """
    # deprecated
    # intel_model = cache.get("latest_intel_model_trans").decode("utf-8")
    sent_model = latest_intel_model_sent.value
    return {
        "sentence_models": sent_model,
        # "model_name": intel_model,
    }


@router.get("/download", status_code=200)
async def download(response: Response):
    """download - downloads dependencies from s3
    Args:
        model: str
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
        model: str
    Returns:
    """
    models = []
    try:
        logger.info("Attempting to download dependencies from S3")
        s3_path = "gamechanger/models/"
        if function == "models":
            models = utils.get_models_list(s3_path)
    except:
        logger.warning(f"Could not get dependencies from S3")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return models

@router.post("/reloadModels", status_code=200)
async def reload_models(model_dict: dict, response: Response):
    """load_latest_models - endpoint for updating the transformer model
    Args:
        model_dict: dict; {"sentence": "bert...", "qexp": "bert...", "transformer": "bert..."}

        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
    """
    model_path_dict = get_model_paths()
    if "sentence" in model_dict:
        SENT_INDEX_PATH = os.path.join(
            Config.LOCAL_PACKAGED_MODELS_DIR, model_dict["sentence"]
        )
        model_path_dict["sentence"] = SENT_INDEX_PATH
    if "qexp" in model_dict:
        QEXP_MODEL_NAME = os.path.join(
            Config.LOCAL_PACKAGED_MODELS_DIR, model_dict["qexp"]
        )
        model_path_dict["qexp"] = QEXP_MODEL_NAME

    logger.info("Attempting to load QE")
    await initQE(model_path_dict["qexp"])
    logger.info("Attempting to load QA")
    await initQA()
    logger.info("Attempting to load Sentence Transformer")
    await initSentence(
        index_path=model_path_dict["sentence"],
        transformer_path=model_path_dict["transformers"],
    )

    logger.info("Reload Complete")
    return

## Post Methods ##

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
        utils.get_s3_corpus(corpus_dict["corpus"])
        # get_transformers(overwrite=False)
        # get_sentence_index(overwrite=False)
    except:
        logger.warning(f"Could not get dependencies from S3")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return

@router.post("/train_model", status_code=200)
async def tain_model(model_dict: dict, response: Response):
    """load_latest_models - endpoint for updating the transformer model
    Args:
        model_dict: dict; {"sentence": "bert...", "qexp": "bert...", "transformer": "bert..."}

        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
    """
    try:
        logger.info("Attempting to download corpus from S3")
        output = subprocess.call(
            ["gamechangerml/scripts/download_corpus.sh"])
        # get_transformers(overwrite=False)
        # get_sentence_index(overwrite=False)
    except:
        logger.warning(f"Could not get dependencies from S3")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return
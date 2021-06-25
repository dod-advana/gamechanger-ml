from fastapi import APIRouter
from fastapi_utils.tasks import repeat_every
import os
from gamechangerml.src.search.QA.QAReader import DocumentReader as QAReader
from gamechangerml.configs.config import QAConfig
from gamechangerml.src.search.query_expansion import qe
from gamechangerml.src.search.sent_transformer.model import SentenceSearcher
from gamechangerml.api.fastapi.settings import *

router = APIRouter()

@router.on_event("startup")
async def initQA():
    """initQA - loads transformer model on start
    Args:
    Returns:
    """
    try:
        global qa_model
        qa_model_path = os.path.join(
            LOCAL_TRANSFORMERS_DIR, "bert-base-cased-squad2")
        logger.info("Starting QA pipeline")
        qa_model = QAReader(qa_model_path, use_gpu=True, **QAConfig.MODEL_ARGS)
        # set cache variable defined in settings.py
        latest_qa_model.value = qa_model_path
        logger.info("Finished loading QA Reader")
    except OSError:
        logger.error(f"Could not load Question Answer Model")


@router.on_event("startup")
async def initQE(qexp_model_path=QEXP_MODEL_NAME):
    """initQE - loads QE model on start
    Args:
    Returns:
    """
    logger.info(f"Loading Query Expansion Model from {qexp_model_path}")
    global query_expander
    try:
        query_expander = qe.QE(
            qexp_model_path, method="emb", vocab_file="word-freq-corpus-20201101.txt"
        )
        logger.info("** Loaded Query Expansion Model")
    except Exception as e:
        logger.warning("** Could not load QE model")
        logger.warning(e)


# Currently deprecated


@router.on_event("startup")
async def initTrans():
    """initTrans - loads transformer model on start
    Args:
    Returns:
    """
    try:
        global sparse_reader
        global latest_intel_model
        model_name = os.path.join(
            LOCAL_TRANSFORMERS_DIR, "distilbert-base-uncased-distilled-squad"
        )
        # not loading due to ram and deprecation
        # logger.info(f"Attempting to load in BERT model default: {model_name}")
        logger.info(
            f"SKIPPING LOADING OF TRANSFORMER MODEL FOR INTELLIGENT SEARCH: {model_name}"
        )
        # sparse_reader = sparse.SparseReader(model_name=model_name)
        latest_intel_model = model_name
        # logger.info(
        #    f" ** Successfully loaded BERT model default: {model_name}")
        logger.info(f" ** Setting Redis model to {model_name}")
        # set cache variable defined in settings.py
        latest_intel_model_trans.value = latest_intel_model
    except OSError:
        logger.error(f"Could not load BERT Model {model_name}")
        logger.error(
            "Check if BERT cache is in correct location: tranformer_cache/ above root directory."
        )


@router.on_event("startup")
async def initSentence(
    index_path=SENT_INDEX_PATH, transformer_path=LOCAL_TRANSFORMERS_DIR
):
    """
    initQE - loads Sentence Transformers on start
    Args:
    Returns:
    """
    global sentence_trans
    # load defaults
    encoder_model = os.path.join(
        transformer_path, "msmarco-distilbert-base-v2")
    logger.info(f"Using {encoder_model} for sentence transformer")
    sim_model = os.path.join(transformer_path, "distilbart-mnli-12-3")
    logger.info(f"Loading Sentence Transformer from {sim_model}")
    logger.info(f"Loading Sentence Index from {index_path}")
    try:
        sentence_trans = SentenceSearcher(
            index_path=index_path,
            sim_model=sim_model,
        )
        # set cache variable defined in settings.py
        latest_intel_model_sent.value  = {"encoder": encoder_model, "sim": sim_model}
        logger.info("** Loaded Sentence Transformers")
    except Exception as e:
        logger.warning("** Could not load Sentence Transformer model")
        logger.warning(e)


@router.on_event("startup")
@repeat_every(seconds=120, wait_first=True)
async def check_health():
    """check_health - periodically checks redis for a new model for workers, checks access to end points
    Args:
    Returns:
    """
    logger.info("API Health Check")
    try:
        new_trans_model_name = str(
            latest_intel_model_trans.value.decode("utf-8")
        )
        new_sent_model_name = str(latest_intel_model_sent.value)
        new_qa_model_name = str(latest_qa_model.value.decode("utf-8"))
    except Exception as e:
        logger.info("Could not get one of the model names from redis")
        logger.info(e)
    try:
        global sparse_reader
        good_health = True
        if (sparse_reader is not None) and (
            sparse_reader.model_name != new_trans_model_name
        ):
            logger.info(
                f"Process does not have {new_trans_model_name} loaded - has {sparse_reader.model_name}"
            )
            sparse_reader = sparse.SparseReader(
                model_name=new_trans_model_name)
            logger.info(f"{new_trans_model_name} loaded")
            good_health = False
    except Exception as e:
        logger.info("Model Health: POOR")
        logger.warn(
            f"Model Health: BAD - Error with reloading model {new_trans_model_name}"
        )
    if check_dep_exist:
        good_health = True
    else:
        good_health = False
    if good_health:
        logger.info("Model Health: GOOD")
    else:
        logger.info("Model Health: POOR")

    # logger.info(f"-- Transformer model name: {new_trans_model_name}")
    logger.info(f"-- Sentence Transformer model name: {new_sent_model_name}")
    logger.info(f"-- QE model name: {QEXP_MODEL_NAME}")
    logger.info(f"-- QA model name: {new_qa_model_name}")

def check_dep_exist():
    healthy = True
    if not os.path.isdir(LOCAL_TRANSFORMERS_DIR):
        logger.warning(f"{LOCAL_TRANSFORMERS_DIR} does NOT exist")
        healthy = False

    if not os.path.isdir(SENT_INDEX_PATH):
        logger.warning(f"{SENT_INDEX_PATH} does NOT exist")
        healthy = False

    if not os.path.isdir(QEXP_MODEL_NAME):
        logger.warning(f"{QEXP_MODEL_NAME} does NOT exist")
        healthy = False
    # topics_dir = os.path.join(QEXP_MODEL_NAME, "topic_models/models")
    # if not os.path.isdir(topics_dir):
    #    logger.warning(f"{topics_dir} does NOT exist")
    #    healthy = False

    return healthy
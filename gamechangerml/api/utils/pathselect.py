import os
import logging
from gamechangerml.api.fastapi.model_config import Config

logger = logging.getLogger()


def get_model_paths():
    model_dict = {}
    # QEXP MODEL
    try:
        qexp_names = [
            f
            for f in os.listdir(Config.LOCAL_PACKAGED_MODELS_DIR)
            if ("qexp_" in f) and (all(substr not in f for substr in ["tar","jbook","ngram"]))
        ]
        qexp_names.sort(reverse=True)
        if len(qexp_names) > 0:
            QEXP_MODEL_PATH = os.path.join(
                Config.LOCAL_PACKAGED_MODELS_DIR, qexp_names[0]
            )
        else:
            print("defaulting INDEX_PATH to qexp")
            QEXP_MODEL_PATH = os.path.join(
                Config.LOCAL_PACKAGED_MODELS_DIR, "qexp_20201217"
            )
    except Exception as e:
        logger.error(e)
        logger.info("Cannot get QEXP model path")
        QEXP_MODEL_PATH = "gamechangerml/models/"

    # QEXP JBOOK MODEL
    try:
        qexp_jbook_names = [
            f
            for f in os.listdir(Config.LOCAL_PACKAGED_MODELS_DIR)
            if (all(substr in f for substr in ["qexp_","jbook"])) and (all(substr not in f for substr in ["ngram","tar"]))
        ]
        qexp_jbook_names.sort(reverse=True)
        if len(qexp_jbook_names) > 0:
            QEXP_JBOOK_MODEL_PATH = os.path.join(
                Config.LOCAL_PACKAGED_MODELS_DIR, qexp_jbook_names[0]
            )
        else:
            print("defaulting INDEX_PATH to qexp")
            QEXP_JBOOK_MODEL_PATH = os.path.join(
                Config.LOCAL_PACKAGED_MODELS_DIR, "jbook_qexp_20211029"
            )
    except Exception as e:
        logger.error(e)
        logger.info("Cannot get QEXP JBOOK model path")
        QEXP_JBOOK_MODEL_PATH = "gamechangerml/models/"

    # QEXP NGRAM MODEL
    try:
        qexp_ngram_names = [
            f
            for f in os.listdir(Config.LOCAL_PACKAGED_MODELS_DIR)
            if (all(substr in f for substr in ["qexp_","ngram"])) and (all(substr not in f for substr in ["jbook","tar"]))
        ]
        qexp_ngram_names.sort(reverse=True)
        if len(qexp_ngram_names) > 0:
            QEXP_NGRAM_MODEL_PATH_LIST = ",".join([os.path.join(
                Config.LOCAL_PACKAGED_MODELS_DIR, qexp_ngram_name
            )
            for qexp_ngram_name in qexp_ngram_names])
            print(f"QEXP_NGRAM_MODEL_PATH_LIST: {QEXP_NGRAM_MODEL_PATH_LIST}")
        else:
            print("defaulting INDEX_PATH to n-gram qexp")
            QEXP_NGRAM_MODEL_PATH_LIST = os.path.join(
                Config.LOCAL_PACKAGED_MODELS_DIR, "qexp_ngram_1_3"
            )
    except Exception as e:
        logger.error(e)
        logger.info("Cannot get QEXP NGRAM model path")
        QEXP_NGRAM_MODEL_PATH_LIST = "gamechangerml/models/"

    # TRANSFORMER MODEL PATH
    try:
        LOCAL_TRANSFORMERS_DIR = os.path.join(
            Config.LOCAL_PACKAGED_MODELS_DIR, "transformers"
        )
    except Exception as e:
        logger.error(e)

        logger.info("Cannot get TRANSFORMER model path")
    # WORK SIM MODEL PATH
    try:
        WORD_SIM_MODEL_PATH = os.path.join(
            QEXP_MODEL_PATH,
            "wiki-news-300d-1M.vec"
            # Config.LOCAL_PACKAGED_MODELS_DIR, "crawl-300d-2M.vec",
        )
    except Exception as e:
        logger.error(e)

        logger.info("Cannot get word sim model path")

    # SENTENCE INDEX
    # get largest file name with sent_index prefix (by date)
    try:
        sent_index_name = [
            f
            for f in os.listdir(Config.LOCAL_PACKAGED_MODELS_DIR)
            if ("sent_index" in f) and ("tar" not in f)
        ]
        sent_index_name = [
            f for f in sent_index_name if os.path.isfile(os.path.join(Config.LOCAL_PACKAGED_MODELS_DIR, f, 'config'))
        ]
        sent_index_name.sort(reverse=True)
        if len(sent_index_name) > 0:
            INDEX_PATH = os.path.join(
                Config.LOCAL_PACKAGED_MODELS_DIR, sent_index_name[0]
            )
        else:
            print("defaulting INDEX_PATH to sent_index")
            INDEX_PATH = os.path.join(
                Config.LOCAL_PACKAGED_MODELS_DIR, "sent_index")
    except Exception as e:
        logger.error(e)
        INDEX_PATH = "gamechangerml/models/"
        logger.info("Cannot get Sentence Index model path")
    model_dict = {
        "transformers": LOCAL_TRANSFORMERS_DIR,
        "sentence": INDEX_PATH,
        "qexp": QEXP_MODEL_PATH,
        "qexp_jbook": QEXP_JBOOK_MODEL_PATH,
        "qexp_ngram": QEXP_NGRAM_MODEL_PATH_LIST,
        "word_sim": WORD_SIM_MODEL_PATH,
    }
    return model_dict

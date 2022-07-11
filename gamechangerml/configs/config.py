from datetime import datetime
import os
from gamechangerml import DATA_PATH, MODEL_PATH


class DefaultConfig:

    DATA_DIR = DATA_PATH
    LOCAL_MODEL_DIR = MODEL_PATH
    DEFAULT_FILE_PREFIX = datetime.now().strftime("%Y%m%d")


class TopicsConfig:

    # topic models should be in folders named gamechangerml/models/topic_model_<date>
    # this path will look for bigrams.phr, tfidf.model, tfidf_dictionary.dic in gamechangerml/models folder as a last resort
    DATA_ARGS = {"LOCAL_MODEL_DIR": MODEL_PATH}

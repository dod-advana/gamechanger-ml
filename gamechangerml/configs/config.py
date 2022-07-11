from datetime import datetime
import os
from gamechangerml import DATA_PATH, MODEL_PATH


class DefaultConfig:

    DATA_DIR = DATA_PATH
    LOCAL_MODEL_DIR = MODEL_PATH
    DEFAULT_FILE_PREFIX = datetime.now().strftime("%Y%m%d")

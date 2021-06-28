import os

from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.api.utils.logger import logger
from gamechangerml.api.utils.redisdriver import *

# get environ vars
GC_ML_HOST = os.environ.get("GC_ML_HOST", default="localhost")
if GC_ML_HOST == "":
    GC_ML_HOST = "localhost"
ignore_files = ["._.DS_Store", ".DS_Store", "index"]

# Redis Cache Variables
latest_intel_model_sent = CacheVariable("latest_intel_model_sent", True)
latest_qa_model = CacheVariable("latest_qa_model")
latest_intel_model_trans = CacheVariable("latest_intel_model_trans")

LOCAL_TRANSFORMERS_DIR = CacheVariable("LOCAL_TRANSFORMERS_DIR")
SENT_INDEX_PATH = CacheVariable("SENT_INDEX_PATH")
QEXP_MODEL_NAME = CacheVariable("QEXP_MODEL_NAME")

model_path_dict = get_model_paths()
LOCAL_TRANSFORMERS_DIR.value = model_path_dict["transformers"]
SENT_INDEX_PATH.value = model_path_dict["sentence"]
QEXP_MODEL_NAME.value = model_path_dict["qexp"]

print(LOCAL_TRANSFORMERS_DIR.value)
t_list = []
try:
    t_list = [trans for trans in os.listdir(
        LOCAL_TRANSFORMERS_DIR.value) if "." not in trans]
except Exception as e:
    logger.warning("No transformers folder")
    logger.warning(e)
logger.info(f"API AVAILABLE TRANSFORMERS are: {t_list}")


# validate correct configurations
logger.info(f"API TRANSFORMERS DIRECTORY is: {LOCAL_TRANSFORMERS_DIR.value}")
logger.info(f"API INDEX PATH is: {SENT_INDEX_PATH.value}")
logger.info(f"API REDIS HOST is: {REDIS_HOST}")
logger.info(f"API REDIS PORT is: {REDIS_PORT}")

# init globals
query_expander = None
sparse_reader = None
latest_intel_model = None
sentence_trans = None
latest_sentence_models = None
qa_model = None
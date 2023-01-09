from logging import getLogger
from gamechangerml.api.utils.pathselect import get_model_paths


model_path_dict = get_model_paths()
try:
    LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
except:
    LOCAL_TRANSFORMERS_DIR = "gamechangerml/models/transformers"
SENT_INDEX_PATH = model_path_dict["sentence"]


logger = getLogger(__name__)

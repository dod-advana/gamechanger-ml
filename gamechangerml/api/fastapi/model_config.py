import os
from ... import REPO_PATH


class Config:
    # ML MODEL PATH
    LOCAL_PACKAGED_MODELS_DIR = f"{REPO_PATH}/gamechangerml/models"
    # os.environ.get("GC_ML_API_MODEL_PARENT_DIR", default=os.path.join("gamechangerml", "models"))
    S3_MODELS_DIR = "models/v3/"

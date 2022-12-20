from pathlib import Path
from os.path import join
from gamechangerml import DATA_PATH, REPO_PATH, MODEL_PATH

S3_DATA_PATH = "bronze/gamechanger/ml-data"
S3_MODELS_PATH = "bronze/gamechanger/models"

CORPUS_DIR = join(REPO_PATH, "gamechangerml", "corpus")

USER_DATA_DIR = join(DATA_PATH, "user_data")
SEARCH_HISTORY_FILE = join(
    USER_DATA_DIR, "search_history", "SearchPdfMapping.csv"
)

FEATURES_DATA_DIR = join(DATA_PATH, "features")
POPULAR_DOCUMENTS_FILE = Path(join(FEATURES_DATA_DIR, "popular_documents.csv"))
COMBINED_ENTITIES_FILE = Path(join(FEATURES_DATA_DIR, "combined_entities.csv"))
TOPICS_FILE = join(FEATURES_DATA_DIR, "topics_wiki.csv")
ORGS_FILE = join(FEATURES_DATA_DIR, "agencies.csv")
PROD_DATA_FILE = join(
    FEATURES_DATA_DIR, "generated_files", "prod_test_data.csv"
)

DEFAULT_SENT_INDEX = join(MODEL_PATH, "sent_index_20210715")

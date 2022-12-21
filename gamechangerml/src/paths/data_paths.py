from os.path import join
from pathlib import Path
from gamechangerml import DATA_PATH, REPO_PATH, MODEL_PATH

# features
FEATURES_DATA_DIR = join(DATA_PATH, "features")
POPULAR_DOCUMENTS_FILE = Path(join(FEATURES_DATA_DIR, "popular_documents.csv"))
COMBINED_ENTITIES_FILE = Path(join(FEATURES_DATA_DIR, "combined_entities.csv"))
TOPICS_FILE = join(FEATURES_DATA_DIR, "topics_wiki.csv")
ORGS_FILE = join(FEATURES_DATA_DIR, "agencies.csv")
ABBREVIATIONS_FILE = join(FEATURES_DATA_DIR, "abbreviations.json")
ABBREVIATIONS_COUNTS_FILE = join(FEATURES_DATA_DIR, "abbcounts.json")

# features/generated_files
FEATURES_GENERATED_FILES_DIR = join(FEATURES_DATA_DIR, "generated_files")
PROD_DATA_FILE = join(FEATURES_DATA_DIR, "prod_test_data.csv")
CORPUS_META_FILE = join(FEATURES_GENERATED_FILES_DIR, "corpus_meta.csv")

# user_data
USER_DATA_DIR = join(DATA_PATH, "user_data")
SEARCH_HISTORY_FILE = join(
    USER_DATA_DIR, "search_history", "SearchPdfMapping.csv"
)

CORPUS_DIR = join(REPO_PATH, "gamechangerml", "corpus")

DEFAULT_SENT_INDEX = join(MODEL_PATH, "sent_index_20210715")

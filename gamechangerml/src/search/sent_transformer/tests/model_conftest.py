# flake8: noqa
# pylint: skip-file

from os.path import join, dirname, isdir, realpath
from pathlib import Path
import pytest
from gamechangerml.src.search.sent_transformer import (
    SentenceEncoder,
    SentenceSearcher,
)
from gamechangerml import REPO_PATH
from gamechangerml.src.configs import EmbedderConfig
from gamechangerml.api.fastapi.settings import LOCAL_TRANSFORMERS_DIR
from gamechangerml.src.utilities import configure_logger

logger = configure_logger()

try:
    here = dirname(realpath(__file__))
    p = Path(here)
    gc_path = REPO_PATH
    test_data_dir = join(str(p), "test_data")
    test_data_2_dir = join(str(p), "test_data_2")
    test_index_dir = join(str(p), "test_index")

    encoder_model_path = join(
        str(gc_path), "gamechangerml/models/transformers/msmarco-distilbert-base-v2"
    )
    assert isdir(test_data_dir)
    assert isdir(test_index_dir)
except (AttributeError, FileExistsError) as e:
    logger.exception("{}: {}".format(type(e), str(e)), exc_info=True)


@pytest.fixture(scope="session")
def sent_dirs():
    return test_data_dir, test_data_2_dir, test_index_dir


@pytest.fixture(scope="session")
def sent_encoder():
    return SentenceEncoder(
        join(LOCAL_TRANSFORMERS_DIR.value, EmbedderConfig.BASE_MODEL)
    )

@pytest.fixture(scope="session")
def sent_searcher():
    return SentenceSearcher(test_index_dir, encoder_model_path)


@pytest.fixture(scope="session")
def topn():
    return 10


@pytest.fixture(scope="session")
def index_files():
    return ["config", "data.csv", "doc_ids.txt", "embeddings", "embeddings.npy"]

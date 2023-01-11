import pytest
from _pytest.assertion import truncate
from os.path import join, dirname, realpath, isdir, exists
from os import makedirs
from shutil import rmtree
from pathlib import Path
from gamechangerml import REPO_PATH
from gamechangerml.configs import SemanticSearchConfig
from gamechangerml.src.utilities import configure_logger
from gamechangerml.src.search.semantic_search import SemanticSearch


# to keep failure messages from being cut off
truncate.DEFAULT_MAX_LINES = 9999
truncate.DEFAULT_MAX_CHARS = 9999


@pytest.fixture(scope="session")
def use_gpu():
    return False


@pytest.fixture()
def corpus_size():
    return 82


@pytest.fixture(scope="session")
def test_data_dir():
    path = join(str(Path(dirname(realpath(__file__)))), "test_data")
    if not isdir(path):
        pytest.exit(f"Missing test data directory: `{path}`.", 1)
    return path


@pytest.fixture(scope="session")
def index_dir(test_data_dir):
    path = join(test_data_dir, "test_index")
    if not isdir(path):
        pytest.exit(f"Test index directory is missing: `{path}`")
    return path


@pytest.fixture(scope="session")
def new_index_dir(test_data_dir):
    """Directory to hold files that are created while testing creation of a new
    index."""
    path = join(test_data_dir, "new_index")
    rmtree(path, ignore_errors=True)
    makedirs(path)
    yield path
    # after tests, remove created files
    rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def semantic_model_path():
    return join(
        REPO_PATH, "gamechangerml", "models", "transformers", SemanticSearchConfig.BASE_MODEL
    )

@pytest.fixture(scope="session")
def logger():
    return configure_logger(
        __name__,
        "DEBUG",
        None,
        "[%(asctime)s %(levelname)-8s], [%(filename)s:%(lineno)s - "
        + "%(funcName)s()], %(message)s",
    )


@pytest.fixture(scope="session")
def search_with_new_index(semantic_model_path, new_index_dir, logger, use_gpu):
    """SemanticSearch object with arg load_index_from_file=False, so that we
    can test the creation of a new index."""
    try:
        searcher = SemanticSearch(
            semantic_model_path, new_index_dir, False, logger, use_gpu
        )
    except Exception:
        pytest.exit(
            f"Failed to init SemanticSearch for fixture `search_with_new_index`",
            1,
        )
    return searcher

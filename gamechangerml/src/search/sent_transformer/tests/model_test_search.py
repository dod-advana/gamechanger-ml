import logging
import os

from gamechangerml.src.search.semantic_search import SemanticSearch
from gamechangerml.configs import SemanticSearchConfig
from gamechangerml.src.utilities.utils import get_local_model_prefix
from gamechangerml import MODEL_PATH
from gamechangerml.src.paths import TRANSFORMERS_DIR

logger = logging.getLogger(__name__)


def test_sent_search(sent_dirs, topn):
    """
    Test for performing a search
    """
    test_data_dir, test_data_2_dir, test_index_dir = sent_dirs

    sent_searcher = SemanticSearch(
        os.path.join(TRANSFORMERS_DIR, SemanticSearchConfig.BASE_MODEL),
        test_index_dir,
        True,
        logger,
        SemanticSearchConfig.DEFAULT_THRESHOLD_ARG,
    )
    queries = ["regulation", "Major Automated Information System"]
    for query in queries:
        results = sent_searcher.search(query, topn, False)
        assert len(results) == topn

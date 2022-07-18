import logging
from os.path import join

from gamechangerml.src.search.sent_transformer import SentenceSearcher
from gamechangerml.src.utilities.utils import get_local_model_prefix
from gamechangerml import MODEL_PATH

from gamechangerml.src.configs import PathConfig, SimilarityRankerConfig

logger = logging.getLogger(__name__)


def test_sent_search(sent_dirs, topn):
    """
    Test for performing a search
    """
    test_data_dir, test_data_2_dir, test_index_dir = sent_dirs

    sent_searcher = SentenceSearcher(
        join(MODEL_PATH, get_local_model_prefix("sent_index")[0]),
        join(PathConfig.TRANSFORMER_PATH, SimilarityRankerConfig.BASE_MODEL_NAME)
    )

    queries = ["regulation", "Major Automated Information System"]
    for query in queries:
        results = sent_searcher.search(query, num_results=topn)
        assert len(results) == topn

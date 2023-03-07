import logging
import os

from gamechangerml.src.search.sent_transformer.model import *
from gamechangerml.src.utilities.utils import get_local_model_prefix
from gamechangerml import MODEL_PATH
from gamechangerml.src.paths import TRANSFORMERS_DIR
from gamechangerml.configs import SimilarityConfig

logger = logging.getLogger(__name__)


def test_sent_search(sent_dirs, topn):
    """
    Test for performing a search
    """
    test_data_dir, test_data_2_dir, test_index_dir = sent_dirs

    sent_searcher = SentenceSearcher(
        sim_model_name=SimilarityConfig.BASE_MODEL,
        index_path=os.path.join(
            MODEL_PATH, get_local_model_prefix("sent_index")[0]),
        transformer_path=TRANSFORMERS_DIR,
    )

    queries = ["regulation", "Major Automated Information System"]
    for query in queries:
        results = sent_searcher.search(query, num_results=topn)
        assert len(results) == topn

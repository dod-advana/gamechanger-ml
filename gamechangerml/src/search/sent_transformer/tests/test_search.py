import logging
import os
from pathlib import Path
import pytest

from gamechangerml.src.search.sent_transformer.model import *
from gamechangerml import REPO_PATH
from gamechangerml.api.fastapi.settings import SENT_INDEX_PATH, LOCAL_TRANSFORMERS_DIR
from gamechangerml.configs.config import SimilarityConfig
logger = logging.getLogger(__name__)


def test_sent_search(sent_dirs, topn):
    """
    Test for performing a search
    """
    test_data_dir, test_data_2_dir, test_index_dir = sent_dirs

    sent_searcher = SentenceSearcher(
        sim_model_name=SimilarityConfig.BASE_MODEL,
        index_path=SENT_INDEX_PATH.value,
        transformer_path=LOCAL_TRANSFORMERS_DIR.value
    )

    queries = ["regulation", "Major Automated Information System"]
    for query in queries:
        results = sent_searcher.search(query, num_results=topn)
        assert len(results) == topn

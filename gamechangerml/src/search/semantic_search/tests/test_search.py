"""Test for SemanticSearch.search()."""
import pytest
from gamechangerml.src.search.semantic_search import SemanticSearch


@pytest.fixture(scope="session")
def search_with_loaded_index(semantic_model_path, index_dir, logger, use_gpu):
    """SemanticSearch object with arg load_index_from_file=True to use a
    pre-existing index."""
    try:
        searcher = SemanticSearch(
            semantic_model_path, index_dir, True, logger, use_gpu
        )
    except Exception:
        pytest.exit(
            f"Failed to init SemanticSearch in `search_with_loaded_index` fixture.",
            1,
        )
    return searcher


def test_num_results_from_search(search_with_loaded_index: SemanticSearch):
    """Test the number of results returned by SemanticSearch.search()"""
    queries = ["regulation", "Major Automated Information System"]
    expected_num_results = 2

    for query in queries:
        actual_num_results = len(search_with_loaded_index.search(query, 2, True))
        assert (
            expected_num_results == actual_num_results
        ), f"Failure: incorrect number of results returned from SemanticSearch.search(). Expected: {expected_num_results}. Actual: {actual_num_results}."

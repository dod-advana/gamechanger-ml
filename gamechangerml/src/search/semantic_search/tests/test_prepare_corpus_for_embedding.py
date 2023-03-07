"""Test for SemanticSearch.prepare_corpus_for_embedding()."""
import pytest
from os.path import join, isdir
from gamechangerml.src.search.semantic_search import SemanticSearch


@pytest.fixture()
def input_dir(test_data_dir):
    path = join(test_data_dir, "prepare_corpus_for_embedding")
    if not isdir(path):
        pytest.exit(
            f"Missing test data for test_prepare_corpus_for_embedding(): `{path}`.",
            1,
        )
    return path


def test_prepare_corpus_for_embedding(
    input_dir,
    corpus_size,
    search_with_new_index: SemanticSearch,
):
    """Test for SemanticSearch.prepare_corpus_for_embedding()."""
    corpus = search_with_new_index.prepare_corpus_for_embedding(input_dir)
    actual_size = len(corpus)

    assert (
        actual_size == corpus_size
    ), f"Failure: number of documents in the corpus. Expected: {corpus_size}. Actual: {actual_size}"

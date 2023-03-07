"""Test SemanticSearch.create_embeddings_index()."""

import pytest
from os.path import join, isfile
from json import load as json_load
from gamechangerml.src.search.semantic_search import SemanticSearch


@pytest.fixture()
def corpus(test_data_dir):
    path = join(test_data_dir, "create_embeddings_index", "input.json")
    if not isfile(path):
        pytest.exit(
            f"Missing test data for test_create_embeddings_index(): `{path}`.",
            1,
        )
    with open(path) as f:
        data = json_load(f)
    return data


@pytest.fixture()
def created_files():
    """Names of files that should be created by running
    SemanticSearch.create_embeddings_index() (with arg save_vectors=False)."""
    return ["data.csv", "doc_ids.txt"]


def test_create_embeddings_index(
    corpus,
    new_index_dir,
    corpus_size,
    created_files,
    search_with_new_index: SemanticSearch,
):
    """Test for SemanticSearch.create_embeddings_index()."""
    search_with_new_index.create_embeddings_index(corpus, False)
    new_files = [join(new_index_dir, fn) for fn in created_files]
    for path in new_files:
        assert isfile(
            path
        ), f"Failed to create file during SemanticSearch.create_embeddings_index(): `{path}`"

    num_ids = len(search_with_new_index.embedder.config["ids"])
    assert (
        num_ids == corpus_size
    ), f"Failure: number of IDs in embedder config after SemanticSearch.create_embeddings_index(). Expected: {corpus_size}. Actual: {num_ids}"

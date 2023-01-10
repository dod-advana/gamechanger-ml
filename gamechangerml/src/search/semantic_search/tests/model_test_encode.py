import logging
import os
import pytest

logger = logging.getLogger(__name__)


def test_sent_encode(sent_encoder, sent_dirs, index_files):
    """
    Test for encoding a corpus folder
    """
    data_dir, data_dir_2, index_dir = sent_dirs
    corpus = sent_encoder.prepare_corpus_for_embedding(data_dir)
    sent_encoder.create_embeddings_index(corpus, False)

    for file in index_files:
        fpath = os.path.join(index_dir, file)
        assert os.path.isfile(fpath)

    embedder_ids = sent_encoder.embedder.config["ids"]

    assert len(embedder_ids) == 82


def test_sent_merge(sent_encoder, sent_dirs, index_files):
    """
    Test for encoding new documents
    """
    data_dir, data_dir_2, index_dir = sent_dirs
    corpus = sent_encoder.prepare_corpus_for_embedding(data_dir)
    sent_encoder.create_embeddings_index(corpus, False)

    for file in index_files:
        fpath = os.path.join(index_dir, file)
        assert os.path.isfile(fpath)

    embedder_ids = sent_encoder.embedder.config["ids"]

    assert len(embedder_ids) == 79

import logging
import os
from gamechangerml.src.search.sent_transformer import (
    prepare_corpus_for_encoder,
)


logger = logging.getLogger(__name__)


def test_sent_encode(
    sent_encoder, sent_dirs, index_files, min_token_len, return_id, verbose
):
    """Test for encoding a corpus folder."""
    data_dir, _, index_dir = sent_dirs
    corpus = prepare_corpus_for_encoder(
        data_dir, min_token_len, return_id, verbose, logger
    )
    sent_encoder.build_index(corpus, index_dir, logger)

    for file in index_files:
        fpath = os.path.join(index_dir, file)
        assert os.path.isfile(fpath)

    embedder_ids = sent_encoder.embedder.config["ids"]

    assert len(embedder_ids) == 145


def test_sent_merge(
    sent_encoder, sent_dirs, index_files, min_token_len, return_id, verbose
):
    """Test for encoding new documents."""
    _, data_dir_2, index_dir = sent_dirs

    corpus = prepare_corpus_for_encoder(
        data_dir_2, min_token_len, return_id, verbose, logger
    )
    sent_encoder.build_index(corpus, index_dir, logger)

    for file in index_files:
        fpath = os.path.join(index_dir, file)
        assert os.path.isfile(fpath)

    embedder_ids = sent_encoder.embedder.config["ids"]

    assert len(embedder_ids) == 271

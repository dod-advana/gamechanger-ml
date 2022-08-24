import logging
from gamechangerml.src.search.query_expansion.utils import remove_original_kw
# flake8: noqa
# pylint: skip-file

import logging
import os
from pathlib import Path

import pytest

from gamechangerml.src.search.query_expansion.build_ann_cli.build_qe_model import (  # noqa
    main,
)
from gamechangerml.src.search.query_expansion.qe import QE
from gamechangerml.configs import QexpConfig
from gamechangerml.api.fastapi.settings import QEXP_MODEL_NAME
log_fmt = (
    "[%(asctime)s %(levelname)-8s], [%(filename)s:%(lineno)s - "
    + "%(funcName)s()], %(message)s"
)
logging.basicConfig(level=logging.DEBUG, format=log_fmt)
logger = logging.getLogger(__name__)

try:
    here = os.path.dirname(os.path.realpath(__file__))
    p = Path(here)
    test_data_dir = os.path.join(p.parents[3], "data", "test_data")
    aux_path = os.path.join(p.parents[3], "data", "features")
    word_wt = os.path.join(aux_path, "enwiki_vocab_min200.txt")
    assert os.path.isfile(word_wt)
except (AttributeError, FileExistsError) as e:
    logger.exception("{}: {}".format(type(e), str(e)), exc_info=True)


@pytest.fixture(scope="session")
def ann_index_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    return str(fn)


@pytest.fixture(scope="session")
def qe_obj(ann_index_dir):
    # main(test_data_dir, ann_index_dir, weight_file=word_wt)
    return QE(
        QEXP_MODEL_NAME.value, **QexpConfig.INIT_ARGS
    )


# @pytest.fixture(scope="session")
# def qe_mlm_obj():
#     return QE(QEXP_MODEL_NAME.value, QexpConfig.INIT_ARGS["qe_files_dir"], "mlm")


@pytest.fixture(scope="session")
def topn():
    return 2


logger = logging.getLogger(__name__)


def check(expanded, exp_len):
    return 1 <= len(expanded) <= exp_len


def test_qe_emb_expand(qe_obj, topn):
    q_str = "security clearance"
    exp = qe_obj.expand(q_str, topn=topn, threshold=0.2, min_tokens=3)
    logger.info(exp)
    assert check(exp, topn)


def test_qe_emb_empty(qe_obj, topn):
    q_str = ""
    exp = qe_obj.expand(q_str, topn=topn, threshold=0.2, min_tokens=3)
    assert len(exp) == 0


def test_qe_emb_oov_1(qe_obj, topn):
    q_str = "kljljfalj"
    exp = qe_obj.expand(q_str, topn=topn, threshold=0.2, min_tokens=3)
    assert len(exp) == 0


def test_qe_emb_iv_2(qe_obj, topn):
    q_str = "financial reporting"
    exp = qe_obj.expand(q_str, topn=topn, threshold=0.2, min_tokens=3)
    assert check(exp, topn)


# @pytest.mark.parametrize(
#     "args",
#     [
#         ["passport", []],
#         [
#             "Find a book, painting, or work of art created in Santa Monica or on the west coast",
#             ["sculpture", "piece"],
#         ],  # noqa
#         ["telework policy for remote work", []],
#         ["telework policy work", ["public"]],
#     ],
# )
# def test_qe_mlm(topn, qe_mlm_obj, args):
#     query, expected = args
#     actual = qe_mlm_obj.expand(query, topn=topn, threshold=0.2, min_tokens=3)
#     assert actual == expected

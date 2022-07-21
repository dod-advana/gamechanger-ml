import json
import logging
import os

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def rank_obj():
    from gamechangerml.src.featurization.rank_features.rank import Rank

    return Rank()


@pytest.fixture
def search_data_sem():
    here = os.path.dirname(os.path.realpath(__file__))
    test_data = os.path.join(here, "sem_test.json")

    with open(test_data) as f:
        resp = json.load(f)
    return resp

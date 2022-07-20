import unittest
from os import makedirs
from os.path import join, exists
from shutil import rmtree
from logging import getLogger
from pandas import DataFrame
from gamechangerml.api.fastapi.settings import LOCAL_TRANSFORMERS_DIR
from gamechangerml.src.configs.embedder_config import EmbedderConfig
from gamechangerml.src.configs.similarity_ranker_config import (
    SimilarityRankerConfig,
)
from gamechangerml.src.search.sent_transformer import (
    SimilarityRanker,
    SentenceEncoder,
    SentenceSearcher,
)
from gamechangerml.src.model_testing.validation_data import MSMarcoData
from ..utils import verify_attribute_exists, verify_attribute_type


class SentenceSearcherTest(unittest.TestCase):
    """Tests for SentenceSearcher.

    Note that these tests are not truly independent, because their setup
    depends on proper functionality of SentenceEncoder.
    """

    @classmethod
    def setUpClass(cls):
        cls.index_path = "./tmp_index"
        if exists(cls.index_path):
            rmtree(cls.index_path)
        makedirs(cls.index_path)

        cls.setup_fail_msgs = []

        try:
            cls.model_path = join(
                LOCAL_TRANSFORMERS_DIR.value,
                SimilarityRankerConfig.BASE_MODEL_NAME,
            )
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to create model path. Exception: {e}."
            )

        try:
            # Make test index
            cls.encoder = SentenceEncoder(
                join(
                    LOCAL_TRANSFORMERS_DIR.value,
                    EmbedderConfig.BASE_MODEL,
                )
            )
            cls.encoder.build_index(
                MSMarcoData().corpus[:5], cls.index_path, getLogger, False
            )
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to create test index. Exception: {e}."
            )

        try:
            cls.searcher = SentenceSearcher(cls.index_path, cls.model_path)
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to init SentenceSearcher. Exception: {e}."
            )

    def setUp(self):
        if self.setup_fail_msgs:
            self.fail(f"Setup failed. {self.setup_fail_msgs}")

    def test_attribute_embedder(self):
        """Verifies the `embedder` attribute of SentenceSearcher."""
        verify_attribute_exists(self, self.searcher, "embedder")

    def test_attribute_data(self):
        """Verifies the `data` attribute of SentenceSearcher."""
        verify_attribute_exists(self, self.searcher, "data")
        verify_attribute_type(self, self.searcher, "data", DataFrame)

    def test_attribute_auto_threshold(self):
        """Verifies the `auto_threshold` attribute of SentenceSearcher."""
        verify_attribute_exists(self, self.searcher, "auto_threshold")
        verify_attribute_type(self, self.searcher, "auto_threshold", float)

    def test_attribute_similarity(self):
        """Verifies the `similarity` attribute of SentenceSearcher."""
        verify_attribute_exists(self, self.searcher, "similarity")
        verify_attribute_type(
            self, self.searcher, "similarity", SimilarityRanker
        )

    def test_retrieve_top_n_with_sim_ranker(self):
        """Verifies SentenceSearcher.retrieve_topn() when the parameter
        use_sim_ranker is True.
        """
        try:
            top_n = self.searcher.retrieve_topn("hello", 3, True)
        except Exception as e:
            self.fail(f"Failed to execute retrieve_top_n(). Exception: {e}.")

        # Verify length of result.
        num_results = len(top_n)
        self.assertTrue(
            num_results == 3,
            f"Result of `retrieve_top_n()` should have length 3 but has length "
            f"{num_results}.",
        )
        # Verify types and keys in result.
        self.assertTrue(
            all(
                [
                    type(x) == dict
                    and set(x.keys()) == {"id", "text", "score"}
                    and type(x["score"]) == float
                    for x in top_n
                ]
            ),
            "Result of `retrieve_top_n()` has incorrect type.",
        )
        # Verify that items are sorted by score (descending).
        is_descending = True
        for i in range(1, num_results):
            if top_n[i]["score"] < top_n[i - 1]["score"]:
                is_descending = False
                break
        if not is_descending:
            self.fail("`retrieve_top_n()` failed to sort results by score.")

    def test_retrieve_top_n_without_sim_ranker(self):
        """Verifies SentenceSearcher.retrieve_topn() when the parameter
        use_sim_ranker is True.
        """
        try:
            top_n = self.searcher.retrieve_topn("hello", 3, True)
        except Exception as e:
            self.fail(f"Failed to execute retrieve_top_n(). Exception: {e}.")

        # Verify keys in results.
        self.assertTrue(
            all(
                [
                    type(x) == dict
                    and set(x.keys())
                    == {"id", "text", "score", "text_length", "passing_result"}
                    for x in top_n
                ]
            )
        )
    
    @classmethod
    def tearDownClass(cls):
        rmtree(cls.index_path)


if __name__ == "__main__":
    unittest.main(failfast=True)

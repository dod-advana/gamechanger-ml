import unittest
from os.path import join
from txtai.pipeline import Similarity
from gamechangerml.src.configs import SimilarityRankerConfig
from gamechangerml.api.fastapi.settings import LOCAL_TRANSFORMERS_DIR
from gamechangerml.src.search.sent_transformer import SimilarityRanker
from ..utils import verify_attribute_exists, verify_attribute_type


class SimilarityRankerTest(unittest.TestCase):
    """Tests for SimilarityRanker."""

    @classmethod
    def setUpClass(cls):
        cls.setup_fail_msgs = []

        try:
            model_path = join(
                LOCAL_TRANSFORMERS_DIR.value,
                SimilarityRankerConfig.BASE_MODEL_NAME,
            )
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to create model path. Exception: {e}."
            )

        try:
            cls.ranker = SimilarityRanker(model_path)
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to init Similarity Ranker. Exception: {e}."
            )

    def setUp(self):
        if self.setup_fail_msgs:
            self.fail(f"Setup failed. {self.setup_fail_msgs}.")

    def test_attribute_exists_model_path(self):
        """Verifies that the `model_path` attribute of SimilarityRanker exists."""
        verify_attribute_exists(self, self.ranker, "model_path")

    def test_attribute_type_model_path(self):
        """Verifies the type of the `model_path` attribute of SimilarityRanker."""
        verify_attribute_type(self, self.ranker, "model_path", str)

    def test_attribute_exists_model(self):
        """Verifies that the `model` attribute of SimilarityRanker exists."""
        verify_attribute_exists(self, self.ranker, "model")

    def test_attribute_type_model(self):
        """Verifies the type of the `model` attribute of SimilarityRanker."""
        verify_attribute_type(self, self.ranker, "model", Similarity)

    def test_rank(self):
        """Verifies SimilarityRanker.rank()."""
        input_query = "hello"
        input_texts = ["hello my name is", "hey there"]

        try:
            result = self.ranker.rank(input_query, input_texts)
        except Exception as e:
            self.fail(
                f"Error occurred when calling SimilarityRanker.rank(). "
                f"Exception: {e}."
            )
        else:
            self.assertTrue(
                len(result) == len(input_texts),
                "Length of result does not equal length of input.",
            )
            self.assertTrue(
                all([type(x) == tuple and len(x) == 2 for x in result]),
                "List item is not tuple or does not have length 2.",
            )
            self.assertTrue(
                all(
                    [type(x[0]) == int and type(x[1]) == float for x in result]
                ),
                "Tuple item does not have correct type.",
            )


if __name__ == "__main__":
    unittest.main(failfast=True)

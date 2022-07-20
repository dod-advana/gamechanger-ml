import unittest
import click
from logging import getLogger
from txtai.embeddings import Embeddings
from txtai.ann import Faiss
from pandas import read_csv
from os.path import join, isdir, isfile
from os import makedirs, listdir
from shutil import rmtree
from gamechangerml.api.fastapi.settings import LOCAL_TRANSFORMERS_DIR
from gamechangerml.src.configs import EmbedderConfig
from gamechangerml.src.model_testing.validation_data import MSMarcoData
from gamechangerml.src.search.sent_transformer import SentenceEncoder
from ..utils import verify_attribute_exists, verify_attribute_type


class SentenceEncoderTest(unittest.TestCase):
    """Tests for SentenceEncoder."""

    @classmethod
    def setUpClass(cls):
        cls.setup_fail_msgs = []

        try:
            model_path = join(
                LOCAL_TRANSFORMERS_DIR.value, EmbedderConfig.BASE_MODEL
            )
        except Exception as e:
            cls.setup_fail_msg.append(
                f"Failed to create model path. Exception: {e}."
            )

        try:
            cls.encoder = SentenceEncoder(model_path)
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to init SentenceEncoder. Exception: {e}."
            )

        try:
            cls.corpus = MSMarcoData().corpus[:5]
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to create test corpus. Exception: {e}."
            )

        try:
            cls.logger = getLogger()
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to get logger. Exception: {e}."
            )

        try:
            cls.tmp_dir = "./tmp"
            if isdir(cls.tmp_dir):
                rmtree(cls.tmp_dir)
            makedirs(cls.tmp_dir)
        except Exception as e:
            cls.setup_fail_msgs.append(
                f"Failed to set up tmp dir. Exception: {e}."
            )

    def setUp(self):
        if self.setup_fail_msgs:
            self.fail(f"Setup failed. {self.setup_fail_msgs}")

    def test_attribute_exists_model_path(self):
        """Verifies that the `model_path` attribute of SentenceEncoder exists."""
        verify_attribute_exists(self, self.encoder, "model_path")

    def test_attribute_type_model_path(self):
        """Verifies that the `model_path` attribute of SentenceEncoder has the
        correct type.
        """
        verify_attribute_type(self, self.encoder, "model_path", str)

    def test_attribute_exists_embedder(self):
        """Verifies that the `embedder` attribute of SentenceEncoder exists."""
        verify_attribute_exists(self, self.encoder, "embedder")

    def test_attribute_type_embedder(self):
        """Verifies that the `model_path` attribute of SentenceEncoder has the
        correct type.
        """
        verify_attribute_type(self, self.encoder, "embedder", Embeddings)

    def test_build_index(self):
        """Verifies SentenceEncoder.build_index()."""
        try:
            self.encoder.build_index(self.corpus, self.tmp_dir, self.logger)
        except Exception as e:
            self.fail(f"Exception occurred when running `build_index()`: {e}.")

        # Verify that data file was created.
        data_path = join(self.tmp_dir, "data.csv")
        self.assertTrue(
            isfile(data_path),
            f"Data csv was not created. No file at path: {data_path}.",
        )

        # Verify that the data file contains expected columns.
        expected_columns = {"text", "paragraph_id"}
        df = read_csv(data_path)
        existing_columns = set(df.columns)
        self.assertEqual(
            existing_columns,
            expected_columns,
            f"Data csv columns are incorrect. Expected columns are: "
            f"{expected_columns}. Actual columns are: {existing_columns}.",
        )

        # Verify that the data file has the expected number of rows.
        num_rows = len(df)
        self.assertEqual(
            5,
            num_rows,
            f"Data csv has {num_rows} rows but is expected to have 5 rows.",
        )

        # Verify that entries in the data file have expected types.
        self.assertTrue(
            all(
                [
                    type(df.loc[i, "paragraph_id"]) == str
                    and df.loc[i, "paragraph_id"].isdigit()
                    for i in range(num_rows)
                ]
            ),
            "Entry in `paragraph_id` column of data csv does not have expected "
            "type.",
        )
        self.assertTrue(
            all([type(df.loc[i, "text"]) == str for i in range(num_rows)]),
            "Entry in `text` column of data csv do not have expected type.",
        )

        # Verify that embedder configs were assigned.
        self.assertIsNotNone(self.encoder.embedder.config.get("ids"))
        self.assertIsNotNone(self.encoder.embedder.config.get("dimensions"))

        # Verify that the embedder's `embeddings` attribute is populated.
        self.assertIsInstance(
            self.encoder.embedder.embeddings, Faiss, "Embeddings not populated"
        )

        # Verify that the embedder was saved.
        self.assertGreater(listdir(self.tmp_dir), 0, "Embedder was not saved.")

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmp_dir)
        

if __name__ == "__main__":
    unittest.main(failfast=True)


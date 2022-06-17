import unittest
from transformers import RobertaTokenizer, Pipeline
from gamechangerml.src.utilities.test_utils import (
    verify_output,
    verify_attribute,
)
from ..section_classifier import SectionClassifier
from ..document_section import DocumentSection


class SectionClassifierTest(unittest.TestCase):
    """Tests for the SectionClassifier class."""

    def setUp(self):
        """Set up for tests."""
        try:
            obj = SectionClassifier()
        except Exception as e:
            self.fail(
                f"Failed to initialize SectionClassifier object. Exception: {e}."
            )
        else:
            data = {"tokenizer": RobertaTokenizer, "pipeline": Pipeline}
            for attr_name, expected_type in data.items():
                verify_attribute(
                    self,
                    obj,
                    attr_name,
                    expected_type,
                    f" Failed to properly initialize SectionClassifier "
                    f"attribute: '{attr_name}'.",
                )
            self.tokenizer = obj.tokenizer
            self.pipeline = obj.pipeline

    def test_chunk_text(self):
        """Verifies SectionClassifier.chunk_text()."""
        fail_msg = "Failed to chunk text."
        try:
            chunks = SectionClassifier.chunk_text("hello", self.tokenizer)
        except Exception as e:
            self.fail(f"{fail_msg} Exception: {e}.")
        else:
            fail_msg += f"chunks: {chunks}"
            self.assertIsInstance(chunks, list, fail_msg)
            is_str = [isinstance(x, str) for x in chunks]
            self.assertNotIn(
                False,
                is_str,
                fail_msg + "Result contains non-string value(s).",
            )

    def test_transform_ids_to_labels(self):
        """Verifies SectionClassifier.transform_ids_to_labels()."""
        fail_msg = "Failed to transform ids to labels."
        try:
            labels = SectionClassifier.transform_ids_to_labels(
                "hello", self.pipeline
            )
        except Exception as e:
            self.fail(f"{fail_msg} Exception: {e}.")
        else:
            fail_msg += f"labels: {labels}"
            self.assertIsInstance(labels, list, fail_msg)
            self.assertTrue(len(labels) > 0, fail_msg)
            self.assertTrue(len(labels[0]) == 3)

    def test_get_sections(self):
        """Verifies SectionClassifier.get_sections()."""
        fail_msg = "Failed to get sections."
        try:
            sections = SectionClassifier.get_sections(
                "hello", self.tokenizer, self.pipeline
            )
        except Exception as e:
            self.fail(f"{fail_msg} Exception: {e}.")
        else:
            fail_msg += f"sections: {sections}"
            self.assertIsInstance(sections, list)
            self.assertTrue(len(sections) > 0)
            is_section = [isinstance(x, DocumentSection) for x in sections]
            self.assertNotIn(False, is_section, fail_msg)

    def test_is_enclosure_line(self):
        """Verifies SectionClassifier.is_enclosure_line."""
        fail_msg = "Failed to determine if text is enclosure line."
        # Keys are input, values are expected output.
        data = {
            "See enclosure 12": True,
            "EnClosure 5": True,
            "hello": False,
            "enclosure": False,
        }
        for input, expected_output in data.items():
            actual_output = SectionClassifier.is_enclosure_line(input)
            verify_output(
                self,
                expected_output,
                actual_output,
                fail_msg + f"input: {input}",
            )


if __name__ == "__main__":
    unittest.main(failfast=True)

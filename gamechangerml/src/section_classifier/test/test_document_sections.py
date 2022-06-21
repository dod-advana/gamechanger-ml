import unittest
from transformers import RobertaTokenizer, pipeline
from gamechangerml.src.section_classifier.document_section import (
    DocumentSection,
)
from gamechangerml.src.utilities.test_utils import verify_attribute
from ..configs import BASE_MODEL_NAME, MODEL_PATH
from ..document_sections import DocumentSections


class DocumentSectionsTest(unittest.TestCase):
    """Tests for the DocumentSections class."""

    def setUp(self):
        """Set up for tests."""
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL_NAME)
            self.pipeline = pipeline(
                "token-classification",
                model=MODEL_PATH,
                tokenizer=self.tokenizer,
            )
            self.obj_populated = DocumentSections(
                record={"pages": [{"p_raw_text": "hi"}]},
                tokenizer=self.tokenizer,
                pipe=self.pipeline,
            )
            self.obj_none = DocumentSections(
                record={}, tokenizer=self.tokenizer, pipe=self.pipeline
            )

        except Exception as e:
            self.fail(f"Setup failed. Exception: {e}.")

    def test_verify_attributes(self):
        """Verifies that the object's attributes were initialized properly."""
        data = {
            "record": dict,
            "all_sections": list,
            "body_sections": list,
            "references_section": str,
        }

        for obj in [self.obj_populated, self.obj_none]:
            for attr_name, expected_type in data.items():
                verify_attribute(
                    self,
                    obj,
                    attr_name,
                    expected_type,
                    f"Failed to properly initialize DocumentSections "
                    f"attribute: '{attr_name}'",
                )

        self.assertTrue(
            len(self.obj_populated.all_sections) > 0
            and isinstance(self.obj_populated.all_sections[0], DocumentSection)
        )


if __name__ == "__main__":
    unittest.main(failfast=True)

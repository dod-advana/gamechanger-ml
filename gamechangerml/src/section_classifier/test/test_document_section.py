import unittest
from gamechangerml.src.utilities.test_utils import verify_attribute
from ..document_section import DocumentSection


class DocumentSectionTest(unittest.TestCase):
    """Tests for the DocumentSection class."""

    def setUp(self):
        try:
            obj = DocumentSection(
                header="\nhello.", text="hi", text_label="Noise", score=0.45
            )
        except Exception as e:
            self.fail(f"Failed to init. Exception: {e}")
        else:
            self.obj = obj

    def test_should_verify_attributes(self):
        """Verifies that attributes were properly created upon object init."""
        data = {"header": str, "text": str, "label": str, "score": float}
        
        for attr_name, expected_type in data.items():
            verify_attribute(
                self,
                self.obj,
                attr_name,
                expected_type,
                f"Failed to properly initialize '{attr_name}' attribute of "
                "DocumentSection.",
            )

    def test_clean_header(self):
        self.assertEqual(self.obj.header, "HELLO", "Failed to clean header.")


if __name__ == "__main__":
    unittest.main(failfast=True)

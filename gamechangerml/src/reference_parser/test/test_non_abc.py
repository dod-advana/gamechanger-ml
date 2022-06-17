import unittest
from gamechangerml.src.utilities.test_utils import verify_output
from ..non_abc.utils import clean_ref_line


class ReferenceParserNonABCTest(unittest.TestCase):
    """Tests for gamechangerml/src/reference_parser/non_abc/"""

    def test_clean_ref_line(self):
        """Verifies clean_ref_line()."""
        fail_msg = "Failed case for clean_ref_line()."
        data = {
            ["", "DoDD 5118.03 CH 1.pdf"]: None,
            ["  \n", "DoDI 7730.64.pdf"]: None,
            ["DoDI 6310.09 enclosure 2", "DoDI 6310.09.pdf"]: None,
            ["references 2", "DoDI 8440.01.pdf"]: None,
            [
                " Executive Order \n\n12333 ",
                "DoDI 1400.25 Volume 731.pdf",
            ]: "Executive Order 12333",
            ["DoD Instruction 8115.02", "DoDI 5500.14.pdf"]: "DoD Instruction "
            "8115.02",
        }
        for inputs, expected_output in data.items():
            text = inputs[0]
            filename = inputs[1]
            actual_output = clean_ref_line(text, filename)
            verify_output(
                self,
                expected_output,
                actual_output,
                fail_msg + f" Inputs: {str(inputs)}",
            )


if __name__ == "__main__":
    unittest.main(failfast=True)

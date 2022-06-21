import unittest
from gamechangerml.src.utilities.test_utils import verify_func, verify_output
from ..shared import is_abc_format, join_by_start_char


class ReferenceParserSharedTest(unittest.TestCase):
    """Tests for gamechangerml/src/reference_parser/shared"""

    def test_is_abc_format(self):
        fail_msg = "Failed case for is_abc_format()."
        data = {
            " \n(a)    DoD Directive 5136.01, “Assistant Secretary of Defense "
            "for Health Affairs (ASD(HA)),”  \n \n  ": True,
            "Deputy Secretary of Defense Memorandum, “Disestablishment of the"
            " Chief Management Officer \nof the DoD and Realignment of "
            "Functions and Responsibilities,” January 11, 2021 ": False,
        }
        verify_func(self, is_abc_format, data, fail_msg)

    def test_join_by_start_char(self):
        """Verifies join_by_start_char()."""
        fail_msg = "Failed case for join_by_commas()."
        data = {
            (
                [
                    "Disestablishment of the Chief Management Officer ",
                    "\tof the DoD and Realignment of Functions and Responsibilities",
                ],
                "\t",
            ): [
                "Disestablishment of the Chief Management Officer of the DoD"
                " and Realignment of Functions and Responsibilities"
            ],
            (["Internal Information Collections", ", June 30, 2014"], ","): [
                "Internal Information Collections, June 30, 2014"
            ],
        }
        for inputs, expected_output in data.items():
            seq = inputs[0]
            char = inputs[1]
            actual_output = join_by_start_char(seq, char)
            verify_output(self, expected_output, actual_output, fail_msg)


if __name__ == "__main__":
    unittest.main(failfast=True)

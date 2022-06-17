import unittest
from gamechangerml.src.utilities.test_utils import verify_func
from ..abc.utils import rm_see_enclosure, split_by_date_and_kws_abc
from ..abc import split_refs_abc


class ReferenceParserABCTest(unittest.TestCase):
    """Tests for gamechangerml/src/reference_parser/abc/"""

    def test_rm_see_enclosure(self):
        """Verifies rm_see_enclosure()."""
        fail_msg = "Failed case for rm_see_enclosure()."
        data = {
            "through (g), See Enclosure 1 (d) Ref1": "Ref1",
            "see Enclosure 1 (a) RefA": "RefA",
            "see Enclosure 3": "",
        }
        verify_func(self, rm_see_enclosure, data, fail_msg)

    def test_split_by_date_and_kws_abc(self):
        """Verifies split_by_date_and_kws_abc()."""
        fail_msg = "Failed case for split_by_date_and_kws_abc()"
        data = {
            "DoD Instruction 1035.01 of April 4, 2012, as amended (b) 5 CFR "
            "2635.704": [
                "DoD Instruction 1035.01 of April 4, 2012, as amended",
                "5 CFR 2635.704",
            ],
            "DoD Instruction 5200.01 of April 21 2016 (k) SECNAVINST "
            "12250.6B": [
                "DoD Instruction 5200.01 of April 21 2016",
                "SECNAVINST 12250.6B",
            ],
            "DoD Instruction 8582.01 of June 6 2012 (hereby canceled) (q) "
            "SECNAVINST 12713.14": [
                "DoD Instruction 8582.01 of June 6 2012 (hereby canceled)",
                "SECNAVINST 12713.14",
            ],
            "Department of Justice, National Training Standards for Sexual "
            "Assault Medical Forensic Examiners, or current version \nDoD "
            "Manual 8910.01, Volume 2": [
                "Department of Justice, National Training Standards for "
                "Sexual Assault Medical Forensic Examiners, or current version",
                "DoD Manual 8910.01, Volume 2",
            ],
            "Executive Order 12333, “United States Intelligence Activities"
            ",” December 4, 1981, as amended": [
                "Executive Order 12333, “United States Intelligence "
                "Activities,” December 4, 1981, as amended"
            ],
        }
        verify_func(self, split_by_date_and_kws_abc, data, fail_msg)

    def test_split_refs_abc(self):
        """Verifies split_refs_abc()."""
        fail_msg = "Failed case for split_refs_abc()."
        data = {
            "(a) DoD Directive 5144.02, “DoD Chief Information Officer (DoD"
            " CIO),” November 21, 2014 \n(b) DoD Directive 8000.01": [
                "DoD Directive 5144.02, “DoD Chief Information Officer (DoD"
                "CIO),” November 21, 2014\n(c) DoD Instruction 8320.02",
                "DoD Directive 8000.01",
                "(c) DoD Instruction 8320.02",
            ]
        }
        verify_func(self, split_refs_abc, data, fail_msg)


if __name__ == "__main__":
    unittest.main(failfast=True)


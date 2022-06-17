from gamechangerml.src.utilities.date_utils import get_months

"""Patterns to help identify the end of a reference. """

REF_END_KWS_PATTERN = rf"""
    (?:
        (,?\s+as\s+amended\s{{0,}})
        |(,?\s+current\s+edition\s{{0,}})
        |(,?\s+most\s+current\s+edition.{{0,2}}\s{{0,}})
        |(\s+\(?hereby\s+cancelled\)?\s{{0,}})
        |(,?\s+current\s+version\s{{0,}})
        |(,?\s+current\s+volume\s{{0,}})
        |(,?\s+date\s+varies\s+by\s+volume\s{{0,}})
    )
"""

months = "|".join(get_months())
DATE_PATTERN = rf"""
    (?:{months})
    \s+
    \d{{1,2}}
    ,?
    (?:\s+)?
    \d{{4}}
"""


def start_char_join(seq, start_char):
    """If a character is at the beginning of a line, join the line with the 
    previous line.
    
    For example, a comma at the beginning of a line implies that the line is a 
    continuation of the previous line. This function joins those lines.

    Args:
        seq (list of str): List of text lines.
        start_char (str): The character to look for at the beginning of lines.

    Returns:
        list of str: The input list, modified in place.
    """
    i = 0
    end = len(start_char) + 1
    while i < len(seq):
        # A comma at the beginning of a line implies that the line is a
        # continuation of the previous line.
        if i > 0 and "," in seq[i][:end]:
            seq[i - 1] += seq.pop(i)
        i += 1

    return seq


import re
from gamechangerml.src.utilities.text_utils import (
    normalize_whitespace,
    is_text_empty,
)
from ...shared import REF_END_KWS_PATTERN, DATE_PATTERN, start_char_join


def split_by_date_and_kws_non_abc(text):
    """If multiple references exist in the input text and are separated
    by a date or keywords, this function will split them into separate
    references.

    Args:
        text (str)
    
    Returns:
        list of str
    """
    pattern_1 = re.compile(
        rf"{DATE_PATTERN}(.{{0,5}}{REF_END_KWS_PATTERN})?",
        flags=re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )
    matches_1 = [
        match
        for match in re.finditer(pattern_1, text)
        if not text.endswith(match.group())
    ]

    if matches_1:
        split_1 = []
        start = 0
        for match in matches_1:
            split_1.append(text[start : match.end()])
            start = match.end()
        split_1.append(text[start:])
    else:
        split_1 = [text]

    pattern_2 = re.compile(
        rf"{REF_END_KWS_PATTERN}(?!$)",
        flags=re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )

    split_2 = []
    for text in split_1:
        matches_2 = [
            match
            for match in re.finditer(pattern_2, text)
            if not text.endswith(match.group())
        ]
        start = 0
        for match in matches_2:
            split_2.append(text[start : match.end()])
            start = match.end()
        split_2.append(text[start:])

    split_2 = [normalize_whitespace(x) for x in split_2]
    split_2 = list(filter(lambda x: not is_text_empty(x, 2), split_2))
    split_2 = start_char_join(split_2, ",")

    return split_2

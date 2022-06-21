import re
from gamechangerml.src.utilities.text_utils import (
    normalize_whitespace,
    is_text_empty,
)
from ...shared import REF_END_KWS_PATTERN, DATE_PATTERN, join_by_start_char


def split_by_date_and_kws_non_abc(text):
    """If multiple references exist in the input text and are separated
    by a date and/ or certain keywords, this function will split them into 
    separate references.

    Args:
        text (str)
    
    Returns:
        list of str
    """
    # Look for a date, optionally followed by keywords, that indicates the end
    # of a reference.
    date_kw_pattern = re.compile(
        rf"{DATE_PATTERN}(.{{0,5}}{REF_END_KWS_PATTERN})?",
        flags=re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )
    date_kw_matches = [
        match
        for match in re.finditer(date_kw_pattern, text)
        if not text.endswith(match.group())
    ]

    if date_kw_matches:
        split = []
        start = 0
        for match in date_kw_matches:
            split.append(text[start : match.end()])
            start = match.end()
        split.append(text[start:])
    else:
        split = [text]

    kw_pattern = re.compile(
        rf"{REF_END_KWS_PATTERN}(?!$)",
        flags=re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )
    result = []

    for text in split:
        # Look for keywords that commonly indicate the end of a reference.
        kw_matches = [
            match
            for match in re.finditer(kw_pattern, text)
            if not text.endswith(match.group())
        ]
        start = 0
        for match in kw_matches:
            result.append(text[start : match.end()])
            start = match.end()
        result.append(text[start:])

    result = [normalize_whitespace(x) for x in result]
    result = list(filter(lambda x: not is_text_empty(x, 2), result))
    result = join_by_start_char(result, ",")

    return result

import re
from gamechangerml.src.utilities.text_utils import (
    is_text_empty,
    normalize_whitespace,
)
from ...shared import REF_END_KWS_PATTERN, DATE_PATTERN, join_by_start_char


def split_by_date_and_kws_abc(text):
    """If 2 ABC list references exist in the input text and are separated
    by a date or keywords, this function will split them into separate
    references.

    Args:
        text (str)

    Returns:
        str
    """
    # Look for list formatting, e.g. "(a)" within ~15 characters after the date.
    # The ~15 character padding is necessary to account for references that
    # include things like "as amended" "hereby canceled", etc. after the date.
    abc_pattern = r"(\s\([a-z]{1,2}\)\s)"
    pattern1 = re.compile(
        rf"{DATE_PATTERN}.{{0,15}}{abc_pattern}",
        flags=re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )
    match = re.search(pattern1, text)

    if match:
        match_str = match.group()
        abc_len = len(
            re.search(
                rf"{abc_pattern}$", match_str, flags=re.IGNORECASE
            ).group()
        )
        # Get rid of ABC formatting but keep the date.
        split = [
            text[: match.start() + len(match_str) - abc_len],
            text[match.end() :],
        ]
    else:
        # If there's no date match, look for keywords that indicate the end of
        # reference ("as amended", "hereby canceled", etc).
        kw_pattern = rf"""
            {REF_END_KWS_PATTERN}
            \n
        """
        pattern2 = re.compile(
            rf"{DATE_PATTERN}.{{1,5}}{kw_pattern}",
            flags=re.VERBOSE | re.IGNORECASE | re.DOTALL,
        )
        match2 = re.search(pattern2, text)
        if match2:
            split = [text[: match2.end()], text[match2.end() :]]
        else:
            split = [text]

    split = [normalize_whitespace(x) for x in split]
    split = list(filter(lambda x: not is_text_empty(x, 2), split))
    split = join_by_start_char(split, ",")

    return split

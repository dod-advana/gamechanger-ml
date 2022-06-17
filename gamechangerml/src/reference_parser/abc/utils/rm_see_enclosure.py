import re
from gamechangerml.src.utilities.text_utils import normalize_whitespace


def rm_see_enclosure(text):
    """Returns the input string with variants of "See Enclosure" removed from 
    the beginning.

    Examples:
        `through (g), See Enclosure 1 (d) <reference d>` --> <reference d>

        `see Enclosure 1 (a) <reference a>` --> `<reference a>`

        `see Enclosure 1` --> ``

    Args:
        text (str)

    Returns:
        str
    """
    text = normalize_whitespace(text)
    through_part = r"""
        (references)?
        (\s)?
        (\([a-z]{1,2}\)\s)?
        through
        \s
        \(
        [a-z]{1,2}
        \)
        \s?
        (,)?
        (\s)?
    """
    text = re.sub(
        rf"^({through_part})", "", text, flags=re.IGNORECASE | re.VERBOSE
    ).strip()

    see_enclosure_num = r"""
        (see)?
        \s?
        enclosure
        \s?
        [a-z0-9]{0,2}
    """
    text = re.sub(
        rf"^({see_enclosure_num})", "", text, flags=re.IGNORECASE | re.VERBOSE
    ).strip()

    list_format = re.search(r"\([a-z]{1,2}\)\s", text[:7])
    if list_format:
        text = text[list_format.end() :].strip()

    return text

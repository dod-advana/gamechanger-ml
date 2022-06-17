import re
from os.path import splitext
from gamechangerml.src.utilities.text_utils import (
    normalize_whitespace,
    is_text_empty,
)


def clean_ref_line(line, filename):
    """Clean a line of text from a document's References section.

    Returns None if the line should not be included in the document's
    parsed references.

    Args:
        line (str): The line of text.
        filename (str): File name of the document. If it is included in the
            line, returns None.
    
    Returns:
        str or None: If str, the cleaned line. None indicates the line should 
        not be included in the document's parsed references.
    """
    title_pattern = rf"""
        references
        \s{{0,}}
        \d{{0,}}
    """
    if line is None:
        return line
    title_pattern = re.compile(
        rf"{title_pattern}", flags=re.VERBOSE | re.IGNORECASE
    )
    line = normalize_whitespace(line)
    if (
        re.fullmatch(title_pattern, line)
        or splitext(filename)[0] in line
        or is_text_empty(line)
    ):
        return None
    else:
        return line

import re


def is_abc_format(refs_section_text):
    """Returns whether or not the given References section text is formatted
    with (a), (b), (c),...

    Args:
        refs_section (str): The text of a document's References section.
    Returns:
        bool: True if the text has ABC formatting, False otherwise.
    """
    return True if re.search("\([a-z]\)", refs_section_text[:80]) else False

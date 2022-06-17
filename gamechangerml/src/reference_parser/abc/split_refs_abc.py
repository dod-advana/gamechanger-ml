import re
from gamechangerml.src.utilities.text_utils import is_text_empty
from .utils import split_by_date_and_kws_abc, rm_see_enclosure


def split_refs_abc(text):
    """Parse the References section of a document.

    To be used for References sections with (a), (b), (c),... formatting.

    Args:
        text (str): The references section of a document.

    Returns:
        list of str: The parsed references section.
    """
    ABC_LIST_PATTERN = "\n(\s+)?\([a-z]{1,2}\)"
    refs = re.split(ABC_LIST_PATTERN, text)

    if len(refs) > 0:
        ABC_START_PATTERN = "^(\([a-z]{1,2}\)(\s+)?)"
        refs[0] = re.sub(ABC_START_PATTERN, "", refs[0])

    clean_refs = []
    for ref in refs:
        if not is_text_empty(ref):
            ref = split_by_date_and_kws_abc(ref)

            for x in ref:
                x = rm_see_enclosure(x)
                if not is_text_empty(x) and len(x) > 5:
                    clean_refs.append(x)
    return clean_refs

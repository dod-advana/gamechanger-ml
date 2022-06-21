from fuzzywuzzy import fuzz
from gamechangerml.src.utilities.text_utils import normalize_whitespace
from gamechangerml.src.section_classifier import SectionClassifier


def filter_refs(refs_list, refs_section, tokenizer, pipe):
    """Filter a list of references.

    1. Remove items that are None or have <= 5 characters.
    2. Remove items that are not labeled as Section Body by the section 
        classifier model.
    3. Normalize whitespace.

    Args:
        refs_list (list of str): List of references.
        refs_section (str): The References section text.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer that 
                will be used by the pipeline to encode data for the model.
        pipe (transformers.Pipeline): Token classification pipeline.

    Returns:
        list of str: The filtered list of references, with non-body
        sections removed.
    """
    refs_list = list(filter(lambda x: x is not None and len(x) > 5, refs_list))
    sections = SectionClassifier.get_sections(refs_section, tokenizer, pipe)
    non_body_sections = [
        normalize_whitespace(section.text)
        for section in SectionClassifier.get_non_body_sections(sections)
    ]

    filtered = []
    for ref in refs_list:
        close_match = False
        for section in non_body_sections:
            if fuzz.ratio(ref, normalize_whitespace(section)) > 95:
                close_match = True
        if not close_match:
            filtered.append(ref)

    return filtered

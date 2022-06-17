import os
import itertools
from gamechangerml.src.utilities.unique import unique
from ..pdf_to_docx import DocxDocument, PDFDocument, pdf_to_docx
from ..shared import start_char_join
from .utils import clean_ref_line, filter_refs, split_by_date_and_kws_non_abc


def split_refs_non_abc(
    pdf_path, pdf_refs_section_text, tokenizer, pipe, delete_docx=True
):
    """Parse the References section of a document.
    
    To be used for References sections that do not have (a), (b), (c),... 
    format. Parses references by converting the PDF to docx and using its 
    formatting patterns.

    Args:
        pdf_path (str): Path to the PDF document.
        pdf_refs_section_text (str): References section text as determined
            by the section parsing model. Note: We must use this (for now) 
            instead of the text from the docx file, because the section parsing 
            model was trained on different paragraph formatting than what is 
            extracted from the docx files.
        tokenizer (transformers.RobertaTokenizer): _description_
        pipe (transformers.Pipeline): Pre-trained section parsing model
        delete_docx (bool, optional): True to delete the docx document after
            it is used to extract references, False otherwise. Defaults to True.

    Returns:
        list of str: Parsed References section.
    """
    pdf_path = os.path.abspath(pdf_path)
    # Get the page numbers where the References section starts and ends.
    start, end = PDFDocument.get_ref_section_page_nums(pdf_path)
    if start is None or end is None:
        return []

    # Convert pdf pages of the References section to docx and extract the text.
    docx_path = os.path.splitext(pdf_path)[0] + ".docx"
    pdf_to_docx(pdf_path, docx_path, [start, end])
    paragraphs = DocxDocument.get_paragraphs(docx_path)

    references = []
    for paragraph in paragraphs:
        lines = paragraph.split("\n")
        # A tab at the beginning of a line implies that the line is a
        # continuation of the previous line.
        references += start_char_join(lines, "\t")

    fn = os.path.splitext(os.path.basename(pdf_path))[0]
    references = list(map(lambda x: clean_ref_line(x, fn), references))
    splits = [
        split_by_date_and_kws_non_abc(x) for x in references if x is not None
    ]
    references = list(itertools.chain.from_iterable(splits))
    references = list(map(lambda x: clean_ref_line(x, fn), references))
    references = filter_refs(
        references, pdf_refs_section_text, tokenizer, pipe
    )
    references = unique(references)

    if delete_docx:
        os.remove(docx_path)

    return references

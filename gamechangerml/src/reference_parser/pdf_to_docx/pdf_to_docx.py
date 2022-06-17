from pdf2docx import parse


def pdf_to_docx(input_file, output_file=None, pages=None):
    """Convert a pdf file to docx.

    Args:
        input_file (str): Path to the PDF to convert.
        output_file (str or None, optional): If str, the path to save the docx
        file. If None, saves to the same location as input_file, changing 
            ".pdf" to ".docx". Defaults to None.
        pages (list of int or None, optional): If a list of ints, converts
            this range of pages. If None, converts all pages. Defaults to None.
    """
    result = parse(pdf_file=input_file, docx_file=output_file, pages=pages)
    return result

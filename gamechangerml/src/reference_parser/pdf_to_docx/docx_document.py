from docx import Document


class DocxDocument:
    @staticmethod
    def get_paragraphs(docx_path):
        """Read the paragraphs of a docx file.

        Args:
            docx_path (str): Path to the docx file.
        Returns:
            list of str: Paragraphs of the docx file.
        """
        doc = Document(docx_path)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]

        return paragraphs

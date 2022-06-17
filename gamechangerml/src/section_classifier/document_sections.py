import re
from gamechangerml.src.utilities.text_utils import is_text_empty
from .configs import SECTION_BODY, REFERENCES_MIN_SCORE
from .section_classifier import SectionClassifier


class DocumentSections:
    """
    Attributes
        record (dict): JSON representation of the document
        all_sections (DocumentSection[]): All sections labeled by the section
            parsing model.
        body_sections (DocumentSection[]): Sections labeled as body by the
            section parsing model.
        reference_section (str): References section from the document.
    """
    def __init__(self, record, tokenizer, pipe):
        """
        Args:
            record (dict): The document in JSON format.
            tokenizer (transformers.RobertaTokenizer)
            pipe (transformers.pipe): Pre-trained section parsing model.
        """
        self.record = record
        self.all_sections = self.get_all_sections(tokenizer, pipe)
        self.body_sections = self.get_body_sections()
        self.references_section = self.get_references_section()

    def get_all_sections(self, tokenizer, pipe):
        """Parse and classify all sections in the document.

        Args:
            tokenizer (transformers.RobertaTokenizer)
        
        Returns: 
            list of DocumentSection
        """
        pages = self.record.get("pages")
        if pages is None:
            return []

        sections = []
        for page in pages:
            text = page.get("p_raw_text")
            if text is None:
                continue
            sections += SectionClassifier.get_sections(text, tokenizer, pipe)

        return sections

    def get_body_sections(self):
        """Returns all sections from the object's all_sections attribute that 
        were classified as Section Body by the section parsing model.

        Returns:
            list of DocumentSection
        """
        return [
            section
            for section in self.all_sections
            if section.label == SECTION_BODY
            and not is_text_empty(section.text)
        ]

    def get_references_section(self):
        """Get the References section of the document.
        
        Returns:
            str
        """
        return "".join(
            [
                section.text.strip()
                for section in self.get_body_sections_including("reference")
                if section.score >= REFERENCES_MIN_SCORE
                and not is_text_empty(section.text, 2)
                and not SectionClassifier.is_enclosure_line(section.text)
            ]
        )

    def get_body_sections_including(self, header_part):
        """Return Section Body sections that include header_part in their 
        associated header.
        
        Returns:
            list of DocumentSection
        """
        return [
            x
            for x in self.body_sections
            if header_part.lower() in x.header.lower()
        ]

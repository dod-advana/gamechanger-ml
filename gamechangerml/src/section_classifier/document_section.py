import re
from string import punctuation


class DocumentSection:
    """
    Attributes:
        header (str): Section header text
        text (str): Section text
        text_label (str): Section label 
        score (float): Measure of confidence of the label assigned
    """
    def __init__(self, header, text, text_label, score):
        """
        Args:
            header (str): Section header text
            text (str): Section text
            text_label (str): Section label 
            score (float): Measure of confidence of the label assigned
        """
        self.header = header
        self.clean_header()
        self.text = text
        self.label = text_label
        self.score = score

    def clean_header(self):
        """Clean the header for this document section.

        Normalize whitespace, remove formatting, leading & trailing
        punctuation, convert to upper case, etc.

        Modifies the object's header attribute in place.
        """
        # Normalize whitespaces.
        self.header = " ".join(self.header.split()).strip()

        # Remove number list formatting.
        # Starting with: a number, (optional) period, and space(s).
        # e.g., "1. hello" --> "hello"
        self.header = re.sub(r"^(\d(\.)?\s)", "", self.header)

        # Remove punctuation at the start and end.
        self.header = self.header.strip(punctuation).strip()

        # Remove "Part x" at start.
        # e.g., "Part I. hello" --> "hello"
        self.header = re.sub(
            r"^(part\s+[a-z]+(\.?)\s+)", "", self.header, flags=re.IGNORECASE
        ).strip()

        # Standardize case.
        self.header = self.header.upper()

    def as_dict(self):
        """Returns the object as a dictionary.

        Returns:
            dict
        """
        d = self.__dict__
        d["score"] = str(self.score)
        return d

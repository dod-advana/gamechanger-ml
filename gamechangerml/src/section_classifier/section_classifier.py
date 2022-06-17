import re
from transformers import RobertaTokenizer, pipeline
from gamechangerml.src.utilities.text_utils import is_text_empty
from .configs import (
    SECTION_BODY,
    SECTION_HEADER,
    WORD_FIELD,
    ENTITY_GROUP_FIELD,
    SCORE_FIELD,
    AGGREGATION_STRATEGY,
    MAX_TOKENS,
    INPUT_IDS_FIELD,
    BASE_MODEL_NAME,
    MODEL_PATH,
)
from .document_section import DocumentSection


class SectionClassifier:
    """Parse and label document sections.
    
    Attributes:
        tokenizer (transformers.RobertaTokenizer)
        pipeline (transformers.Pipeline): Pre-trained section classifier model.
    """

    """Token Labels dictionary."""
    IDS_TO_LABELS = {
        0: SECTION_HEADER,
        1: SECTION_BODY,
        2: "Intro-Data",
        3: "Noise",
        4: "Poor-Quality",
        -100: -100,
    }

    def __init__(self, base_model_name=BASE_MODEL_NAME, model_path=MODEL_PATH):
        """Parse and label document sections.

        Args:
            base_model_name (str, optional): Name of a predefined tokenizer 
                hosted inside a model repo on huggingface.co. Defaults to 
                BASE_MODEL_NAME.
            model_path (str, optional): Path to the pre-trained section 
                classifier model. Defaults to MODEL_PATH.
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
        self.pipeline = pipeline(
            "token-classification", model=model_path, tokenizer=self.tokenizer
        )

    @staticmethod
    def chunk_text(text, tokenizer, max_tokens=MAX_TOKENS):
        """Chunk text into sections no greater than max_tokens.

        Args:
            text (str): The text to split into chunks.
            tokenizer (transformers.RobertaTokenizer)
            max_tokens (int, optional): Maximum number of tokens per chunk of 
                text.
        Returns:
            list of str: The chunked text.
        """
        encoded = tokenizer(text)[INPUT_IDS_FIELD]
        num_tokens = len(encoded)

        if num_tokens > max_tokens:
            chunks_encoded = [
                encoded[i : i + max_tokens]
                for i in range(0, len(encoded), max_tokens)
            ]
            return [tokenizer.decode(x)[3:-4] for x in chunks_encoded]
        else:
            return [text]

    @staticmethod
    def transform_ids_to_labels(text, pipe):
        """Transform section IDs to labels.

        Args:
            text (str)
            pipe (transformers.Pipeline)
        Returns:
            list of tuples, each of length 3: List of labeled sections. Each 
            tuple contains:
                - str: The label assigned to the section by the section 
                    classifier model
                - str: Section text
                - float: Score
        """
        return [
            (
                SectionClassifier.IDS_TO_LABELS[
                    int(e[ENTITY_GROUP_FIELD][-1])
                ],
                e[WORD_FIELD],
                e[SCORE_FIELD],
            )
            for e in pipe(text, aggregation_strategy=AGGREGATION_STRATEGY)
        ]

    @staticmethod
    def get_sections(text, tokenizer, pipe):
        """Parse a document into labeled sections using the section classifier 
        model.

        Args:
            text (str)
            tokenizer (transformers.RobertaTokenizer)
            pipe (transformers.Pipeline)
        
        Returns: 
            list of DocumentSection[]
        """
        sections = []
        header = "None"
        text = text.replace(" \n", "\n ").replace("\n \n ", "\n")
        text_chunks = SectionClassifier.chunk_text(text, tokenizer)

        for chunk in text_chunks:
            for section in SectionClassifier.transform_ids_to_labels(
                chunk, pipe
            ):
                label = section[0]
                text = section[1]
                score = section[2]
                if label == SECTION_HEADER:
                    header = text.replace("\n", "")
                else:
                    sections.append(
                        DocumentSection(header, text, label, score)
                    )

        return sections

    @staticmethod
    def get_non_body_sections(sections):
        """Return sections that were not labeled as Section Body by the section 
        classifier model.

        Args:
            sections (list of DocumentSection): The sections to filter.

        Returns:
            list of DocumentSection
        """
        return [
            section
            for section in sections
            if section.label != SECTION_BODY
            and not is_text_empty(section.text, 5)
        ]

    @staticmethod
    def is_enclosure_line(text):
        """Returns whether or not the given text is a variant of 'See Enclosure 
        X'.

        Returns:
            bool
        """
        match = re.fullmatch(
            r"(see\s)?enclosure\s[0-9a-z]{1,2}",
            " ".join(text.split()).strip().replace(".", ""),
            flags=re.IGNORECASE,
        )
        return match is not None

from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from statistics import median
from typing import List, Tuple, Union
from tqdm import tqdm
from gamechangerml.src.utilities.file_utils import save_pickle
from random import shuffle
import spacy
from ..bi_encoder_config import BiEncoderConfig
from ..util import BiEncoderPreprocessor


class BiEncoderTrainingData:
    """Transform document passages into training data for a bi-encoder model.

    Args:
        query_generator (gamechangerml.src.search.semantic_search.QueryGenerator):
            The model which creates relevant queries for document passages.
        query_classifier (gamechangerml.src.search.semantic_search.QueryClassifier):
            The model which classifies whether or not the generated queries are
            suitable to use as training data.
        minimum_passage_length (int, optional): The minimum length a passage
            must have (after preprocessing) in order to include it in the
            training data. Defaults to BiEncoderConfig.MINIMUM_PASSAGE_LENGTH.
        minimum_line_length (int, optional): The minimum median line length a
            passage must have in order to include it in the training data. This
            helps weed out headers, footers, etc. Defaults to
            BiEncoderConfig.MINIMUM_LINE_LENGTH.
        spacy_model (str, optional): Spacy model to use to tokenize passages.
            Defaults to BiEncoderConfig.SPACY_MODEL. https://spacy.io/models/en
    """

    def __init__(
        self,
        query_generator,
        query_classifier,
        minimum_passage_length: int = BiEncoderConfig.MINIMUM_PASSAGE_LENGTH,
        minimum_line_length: int = BiEncoderConfig.MINIMUM_LINE_LENGTH,
        spacy_model: str = BiEncoderConfig.SPACY_MODEL,
    ):
        self._minimum_passage_length = minimum_passage_length
        self._query_generator = query_generator
        self._query_classifier = query_classifier
        self._minimum_line_length = minimum_line_length
        self._nlp = spacy.load(spacy_model)

    def create_examples(
        self,
        passages: List[str],
        eval_size: float,
        max_queries_per_passage: int,
        random_state: int = BiEncoderConfig.RANDOM_STATE,
        show_progress: bool = True,
        save_dir: Union[str, None] = None,
        noun_replacement_minimum: Union[
            float, None
        ] = BiEncoderConfig.NOUN_REPLACEMENT_MINIMUM,
        noun_replacement_limit: Union[
            int, None
        ] = BiEncoderConfig.NOUN_REPLACEMENT_LIMIT,
    ) -> Tuple[List[InputExample], List[InputExample]]:
        """Create training examples for a list of passages.

        For each passage, `max_queries_per_passage` relevant queries
        (positive examples) and augmented text (negative examples) will be
        generated.

        Positive examples are created with synthetic query generation.
        Negative examples are created by randomly shuffling noun chunks in the
        passage to change the context/ meaning.

        Args:
            passages (List[str]): The document passages to create queries for
                and transform into training data. Some passages will be weeded
                out if they are not deemed suitable.
            eval_size (float or int): If float, should be between 0.0 and 1.0
                and represent the proportion of the dataset to include in the
                evaluation samples. If int, represents the absolute number of
                evaluation samples.
            max_queries_per_passage (int): The maximum number of queries to
                generate per passage.
            random_state (int, optional): Affects the ordering of the passage
                indices when the train and evaluation samples are split.
                Defaults to BiEncoderConfig.RANDOM_STATE.
            show_progress (bool, optional): True to show a progress bar, False
                otherwise. Defaults to True.
            save_dir (Union[str, None], optional): If str, the directory path
                to save the train and evaluation InputExample lists to, with
                filenames "train_examples.pkl" and "eval_examples.pkl",
                respectively. If None, does not save. Defaults to None.
            noun_replacement_minimum (Union[float, None], optional): The minimum
                ratio (number of noun chunks / number of document tokens)
                required in the augmented passage text for the example to be
                included in the result. Defaults to
                BiEncoderConfig.NOUN_REPLACEMENT_MINIMUM.
            noun_replacement_limit (Union[int, None], optional): The maximum
                number of noun chunks to shuffle in the augmented passage text.
                If None, there is no limit. Defaults to None.

        Returns:
            Tuple[List[InputExample], List[InputExample]]: The train and
                evaluation samples.
        """
        all_examples = []

        if show_progress:
            passages = tqdm(passages)

        for pos_text in passages:
            query_passage_pairs = self._prepare_positive_queries_and_passage(
                pos_text, max_queries_per_passage
            )

            for query_text, pos_text in query_passage_pairs:
                (
                    neg_text,
                    ratio_changed_tokens,
                ) = self._shuffle_noun_chunks(pos_text, noun_replacement_limit)

                if ratio_changed_tokens >= noun_replacement_minimum:
                    all_examples.append(
                        InputExample(texts=[query_text, pos_text, neg_text])
                    )

        train_examples, eval_examples = train_test_split(
            all_examples, test_size=eval_size, random_state=random_state
        )

        if save_dir is not None:
            save_pickle(train_examples, save_dir, "train_examples.pkl")
            save_pickle(eval_examples, save_dir, "eval_examples.pkl")

        return train_examples, eval_examples

    def _shuffle_noun_chunks(self, text: str, limit: Union[int, None]) -> str:
        doc = self._nlp(text)

        all_noun_chunks = list(doc.noun_chunks)
        usable_noun_chunks = list(all_noun_chunks)
        shuffle(usable_noun_chunks)
        start_idxs = [noun.start for noun in all_noun_chunks]

        if limit is None:
            limit = len(all_noun_chunks)

        num_replaced = 0
        augmented_text = ""
        token_idx = 0

        while token_idx < len(doc) and num_replaced < limit:
            token = doc[token_idx]
            if token_idx in start_idxs:
                current_noun = [
                    noun for noun in all_noun_chunks if noun.start == token_idx
                ][0]
                token_idx = current_noun.end
                replacement_noun = usable_noun_chunks.pop(0)
                augmented_text += f" {replacement_noun.text} "
                num_replaced += 1
            else:
                augmented_text += f" {token.text} "
                token_idx += 1

        augmented_text = BiEncoderPreprocessor.process(
            " ".join(augmented_text.split()), True
        )

        return augmented_text, len(all_noun_chunks) / len(doc)

    def _prepare_positive_queries_and_passage(
        self, passage: str, max_queries: int
    ) -> List[List[str]]:
        # Only remove newlines *after* checking _is_passage_acceptable() b/c
        # it checks the median line length which is dependent on newlines.
        passage = BiEncoderPreprocessor.process(passage, False)

        if not self._is_passage_acceptable(passage):
            return []

        passage = " ".join(passage.split()).strip()

        positive_queries = self._query_generator.generate(passage, max_queries)
        positive_queries = [
            query
            for query in positive_queries
            if self._query_classifier.is_query_acceptable(query)
        ]

        return [[query, passage] for query in set(positive_queries)]

    def _is_passage_acceptable(self, text: str) -> bool:
        text_without_newlines = " ".join(text.split())

        if len(text_without_newlines) < self._minimum_passage_length:
            return False
        if "....." in text:  # table of contents
            return False
        if " _ _ _ _" in text:  # signature line
            return False
        if "_______" in text:  # line for handwriting
            return False

        if self._median_line_length(text) < self._minimum_line_length:
            return False

        return True

    def _median_line_length(self, text) -> int:
        lines = text.split("\n")
        lines = [line.strip() for line in lines if line and not line.isspace()]
        lengths = [len(line) for line in lines]

        return median(lengths)

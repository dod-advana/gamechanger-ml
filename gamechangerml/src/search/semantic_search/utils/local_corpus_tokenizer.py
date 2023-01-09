from threading import current_thread
from numpy import median
from re import search
from tqdm import tqdm
from gamechangerml.src.text_handling.process import preprocess
from gamechangerml.src.utilities import get_json_paths_for_directory, open_json
from gamechangerml.api.utils import processmanager


class LocalCorpusTokenizer:
    """Tokenize paragraphs of a (JSON) corpus that was created by the
    gamechanger-data document parser pipeline.

    This will only process the first level of the given directory. Nested
    files will not be processed.

    Args:
        directory_path (str): Path to directory of JSON corpus files.
        min_token_len (int, optional): Defaults to 3.
        files_to_use (list of str or None, optional): List of JSON file names
            to process. If empty, will process all JSON files in the directory.
            Defaults to None.
        median_token_len_threshold (int|float, optional): After preprocessing,
            discard paragraphs with median token length less than this
            threshold. Defaults to 2.5.
        repeat_tokens_threshold (float, optional): Defaults to 0.2. After
            preprocessing, discard paragraphs with ratio of unique tokens less
            than this threshold.
        long_token_len_threshold (int, optional): Defaults to 25. After
            preprocessing, a token with length greater than this threshold is
            considered an extra long token.
        long_token_ratio_threshold (float, optional): Defaults to 0.05. After
            preprocessing, discard a paragraph if its ratio of extra long tokens
            is greater than this threshold.
    """

    def __init__(
        self,
        directory_path,
        files_to_use=None,
        min_num_tokens_per_paragraph=25,
        median_token_len_threshold=2.5,
        repeat_tokens_threshold=0.2,
        long_token_len_threshold=25,
        long_token_ratio_threshold=0.05,
    ):
        self._directory_path = directory_path
        self._file_paths = get_json_paths_for_directory(files_to_use)

        self._min_num_tokens = min_num_tokens_per_paragraph
        self._median_token_len_threshold = median_token_len_threshold
        self._repeat_tokens_threshold = repeat_tokens_threshold
        self._long_token_len_threshold = long_token_len_threshold
        self._long_token_ratio_threshold = long_token_ratio_threshold

        # The number of documents that have been loaded and tokenized.
        self._corpus_load_progress = 0
        # Total number of documents to load and tokenize.
        self._corpus_load_total = len(self._file_paths)

    def __iter__(self):
        self._corpus_load_progress = 0
        self._update_process_manager()

        for file_name in tqdm(self._file_paths):
            try:
                paragraphs = open_json(file_name)["paragraphs"]

                for paragraph in paragraphs:
                    text = paragraph["par_raw_text_t"]
                    id_ = paragraph["id"]
                    tokens = preprocess(text, min_len=1)

                    if len(
                        tokens
                    ) > self._min_num_tokens and self._is_quality_after_preprocess(
                        tokens, text
                    ):
                        yield tokens, id_

                self._corpus_load_progress += 1
                self._update_process_manager()
            except Exception as e:
                print(f"{e}\nError with {file_name} in creating local corpus")

    def _is_quality_after_preprocess(self, processed_tokens, raw_text) -> bool:
        """Returns whether or not the paragraph should be yielded in __iter__."""
        raw_tokens = raw_text.split(" ")

        # Check if most of the tokens were filtered out during preprocessing.
        if len(processed_tokens) / len(raw_tokens) <= 0.5:
            return False

        # Check if the median length of processed tokens is less than the
        # expected threshold.
        median_token_len = median([len(token) for token in processed_tokens])
        if median_token_len <= self._median_token_len_threshold:
            return False

        # Check if the ratio of unique tokens is less than an expected threshold.
        unique_tokens_ratio = len(set(processed_tokens)) / len(raw_tokens)
        if unique_tokens_ratio < self._repeat_tokens_threshold:
            return False

        # Check for a high percentage of long tokens, excluding certain
        # website-related tokens.
        website_tokens = ["http", "www."]
        non_website_raw_tokens = [
            token
            for token in processed_tokens
            if not any(
                [token.startswith(web_token) for web_token in website_tokens]
            )
        ]
        long_tokens = [
            token
            for token in non_website_raw_tokens
            if len(token) > self._long_token_len_threshold
        ]
        if (
            len(long_tokens) / len(processed_tokens)
            > self._long_token_ratio_threshold
        ):
            return False

        # Check if the text appears to be a table of contents.
        if search(r"\.{6,}", raw_text):
            return False

        return True

    def _update_process_manager(self):
        processmanager.update_status(
            processmanager.loading_corpus,
            self._corpus_load_progress,
            self._corpus_load_total,
            thread_id=current_thread().ident,
        )

from json import loads
from threading import current_thread
from os.path import join
from os import listdir
from tqdm import tqdm

from gamechangerml.src.text_handling.process import preprocess, get_tokenizer
from gamechangerml.src.utilities.text_utils import check_quality_paragraph
from gamechangerml.api.utils import processmanager


class LocalCorpusTokenizer(object):
    """Tokenize a local JSON corpus.

    Args:
        directory (str): Path to directory of JSON corpus files.
        return_id (bool, optional): True to return (tokens, paragraph id) items
            in iteration. False to return only tokens. Defaults to False.
        min_token_len (int, optional): Defaults to 3.
        verbose (bool, optional): True to print progress during iteration, False
            otherwise. Defaults to False.
        bert_based_tokenizer (str or None, optional): Path to a bert tokenizer
            to use. If None, uses preprocess() instead of a bert based
            tokenizer. Defaults to None.
        files_to_use (list of str): List of JSON file names to process. If empty,
            will process all JSON files in the directory.
    """

    def __init__(
        self,
        directory,
        return_id=False,
        min_token_len=3,
        verbose=False,
        bert_based_tokenizer=None,
        files_to_use=None,
    ):
        self.directory = directory

        if files_to_use:  ## if we only want to do this on a subset
            self.file_list = list(
                set([join(directory, i) for i in files_to_use]).intersection(
                    [
                        join(directory, file)
                        for file in listdir(directory)
                        if file[-5:] == ".json"
                    ]
                )
            )
        else:
            self.file_list = [
                join(directory, file)
                for file in listdir(directory)
                if file[-5:] == ".json"
            ]

        self.return_id = return_id
        self.min_token_len = min_token_len
        self.verbose = verbose
        self.bert_based_tokenizer = bert_based_tokenizer
        if self.bert_based_tokenizer:
            self.auto_token = get_tokenizer(self.bert_based_tokenizer)

    def __iter__(self):
        if self.verbose:
            iterator = tqdm(self.file_list)
        else:
            iterator = self.file_list

        total = len(self.file_list)
        progress = 0
        processmanager.update_status(
            processmanager.loading_corpus,
            progress,
            total,
            thread_id=current_thread().ident,
        )
        for file_name in iterator:
            try:
                doc = self._get_doc(file_name)
                paragraphs = [p["par_raw_text_t"] for p in doc["paragraphs"]]
                paragraph_ids = [p["id"] for p in doc["paragraphs"]]

                for para_text, para_id in zip(paragraphs, paragraph_ids):
                    if self.bert_based_tokenizer:
                        tokens = self.auto_token.tokenize(para_text)
                    else:
                        tokens = preprocess(para_text, min_len=1)
                    if tokens:
                        if check_quality_paragraph(tokens, para_text):
                            if len(tokens) > self.min_token_len:
                                if self.return_id:
                                    yield tokens, para_id
                                else:
                                    yield tokens

                progress += 1
                processmanager.update_status(
                    processmanager.loading_corpus,
                    progress,
                    total,
                    thread_id=current_thread().ident,
                )
            except Exception as e:
                print(f"{e}\nError with {file_name} in creating local corpus")

    def _get_doc(self, file_name):
        with open(file_name, "r") as f:
            line = f.readline()
            line = loads(line)
        return line

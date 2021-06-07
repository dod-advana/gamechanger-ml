from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

from transformers import BertTokenizer


def preprocess(
    text,
    min_len=2,
    phrase_detector=None,
    remove_stopwords=False,
    additional_stopwords=None,
):
    """
    preprocess - standard text processing (possibly break out more if complex preprocessing needed
    Args:
        text (str)
        min_len (int): optional Minimum length of token (inclusive). Shorter tokens are discarded.
        remove_stopwords (bool)
        additional_stopwords (list of strings)
    Returns:
        tokens (list of strings)
    """
    tokens = simple_preprocess(text, min_len=min_len, max_len=20)

    if phrase_detector != None:
        tokens = phrase_detector.apply(tokens)

    if remove_stopwords:
        if additional_stopwords != None:
            stopwords_list = STOPWORDS.union(set(additional_stopwords))
        else:
            stopwords_list = STOPWORDS
        tokens = [word for word in tokens if word not in stopwords_list]

    return tokens


def topic_processing(text: str, phrase_model: object):
    """
    topic_processing - simple preprocessing model to be used in conjunction with the TF-iDF topic model
    Args:
        text (str)
        phrase_model (object): any model that takes in a tokenized text list as an indexed arg and
            returns a tokenized list with with phrases combined. (See Gensims Phrases/Phraser model).
    """
    tokens = phrase_model[simple_preprocess(text, min_len=4, max_len=15)]
    return tokens


class bert_tokenizer(object):
    def __init__(self, vocab_file=None):

        if vocab_file is None:
            vocab_file = "./assets/bert_vocab.txt"
        self.tokenizer = BertTokenizer(
            vocab_file=vocab_file, do_lower_case=True
        )

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)

        return tokens, len(tokens)

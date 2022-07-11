from datetime import datetime
import os
from gamechangerml import DATA_PATH, MODEL_PATH


class DefaultConfig:

    DATA_DIR = DATA_PATH
    LOCAL_MODEL_DIR = MODEL_PATH
    DEFAULT_FILE_PREFIX = datetime.now().strftime("%Y%m%d")


# for Bert Extractive Summarizer (https://pypi.org/project/bert-extractive-summarizer/)
class BertSummConfig:
    MODEL_ARGS = {
        "initialize": {
            # This gets used by the hugging face bert library to load the model, you can supply a custom trained model here
            "model": "bert-base-uncased",
            # If you have a pre-trained model, you can add the model class here.
            "custom_model": None,
            # If you have a custom tokenizer, you can add the tokenizer here.
            "custom_tokenizer": None,
            # Needs to be negative, but allows you to pick which layer you want the embeddings to come from.
            "hidden": -2,
            # It can be "mean", "median", or "max". This reduces the embedding layer for pooling.
            "reduce_option": "mean",
        },
        "fit": {
            "ratio": None,  # The ratio of sentences that you want for the final summary
            # Parameter to specify to remove sentences that are less than 40 characters
            "min_length": 40,
            "max_length": 600,  # Parameter to specify to remove sentences greater than the max length
            # Number of sentences to use. Overrides ratio if supplied.
            "num_sentences": 2,
        },
        "coreference": {"greedyness": 0.4},
        "doc_limit": 100000,
    }


class TopicsConfig:

    # topic models should be in folders named gamechangerml/models/topic_model_<date>
    # this path will look for bigrams.phr, tfidf.model, tfidf_dictionary.dic in gamechangerml/models folder as a last resort
    DATA_ARGS = {"LOCAL_MODEL_DIR": MODEL_PATH}

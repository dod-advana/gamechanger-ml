from datetime import datetime
from os import environ
import os
from gamechangerml import REPO_PATH

class DefaultConfig:

    DATA_DIR = os.path.join(REPO_PATH, "common/data/processed")
    LOCAL_MODEL_DIR = os.path.join(REPO_PATH, "gamechangerml/models")
    DEFAULT_FILE_PREFIX = datetime.now().strftime("%Y%m%d")
    # DEFAULT_MODEL_NAME = "20200728"
    # MODEL_DIR = "gamechangerml/src/modelzoo/semantic/packaged_models/20200728"
    # LOCAL_PACKAGED_MODELS_DIR = os.path.join(REPO_PATH, "gamechangerml/src/modelzoo/semantic/packaged_models")

class S3Config:
    STORE_S3 = True
    S3_MODELS_DIR = "models/v3/"
    S3_CORPUS_DIR = "corpus/"


class D2VConfig:
    # MODEL_ID = datetime.now().strftime("%Y%m%d")
    # MODEL_DIR = os.path.join(REPO_PATH, "gamechangerml/src/modelzoo/semantic/models")
    # CORPUS_DIR = "../tinytestcorpus"
    # CORPUS_DIR = "test/small_corpus"
    MODEL_ARGS = {
        "dm": 1,
        "dbow_words": 1,
        "vector_size": 50,
        "window": 5,
        "min_count": 5,
        "sample": 1e-5,
        "epochs": 20,
        "alpha": 0.020,
        "min_alpha": 0.005,
        # 'workers': multiprocessing.cpu_count() // 2 # to allow some portion of the cores to perform generator tasks
    }

# for Bert Extractive Summarizer (https://pypi.org/project/bert-extractive-summarizer/)
class BertSummConfig:
    MODEL_ARGS = {
        "initialize": {
            # This gets used by the hugging face bert library to load the model, you can supply a custom trained model here
            "model": 'bert-base-uncased',
            # If you have a pre-trained model, you can add the model class here.
            "custom_model": None,
            # If you have a custom tokenizer, you can add the tokenizer here.
            "custom_tokenizer":  None,
            # Needs to be negative, but allows you to pick which layer you want the embeddings to come from.
            "hidden": -2,
            # It can be 'mean', 'median', or 'max'. This reduces the embedding layer for pooling.
            "reduce_option": 'mean'
        },
        "fit": {
            "ratio": None,  # The ratio of sentences that you want for the final summary
            # Parameter to specify to remove sentences that are less than 40 characters
            "min_length": 40,
            "max_length": 600,  # Parameter to specify to remove sentences greater than the max length
            # Number of sentences to use. Overrides ratio if supplied.
            "num_sentences": 2
        },
        "coreference": {
            "greedyness": 0.4
        },
        "doc_limit": 100000
    }

class QAConfig:
    MODEL_ARGS = {
        "model_name": "bert-base-cased-squad2", # SOURCE:
        "qa_type": 'scored_answer', # options are: ['scored_answer', 'simple_answer']
        "nbest": 1, # number of answers to retrieve from each context for comparison
        "null_threshold": -3 # if diff between the answer score and null answer score is greater than this threshold, don't return answer
    }

class EmbedderConfig: 
    MODEL_ARGS = {
        "model_name": "msmarco-distilbert-base-v2", # SOURCE
        "embeddings": {
            "embeddings": "embeddings.npy",
            "dataframe": "data.csv",
            "ids": "doc_ids.txt",
        },
        "encoder": { ## args for making the embeddings index
            "min_token_len": 10,
            "overwrite": False,
            "verbose": True, # for creating LocalCorpus
            "return_id": True # for creating LocalCorpus
        },
        "retriever": { ## args for retrieving the vectors
            "n_returns": 10
        }
    }

class SimilarityConfig:
    MODEL_ARGS = {
        "model_name": "distilbart-mnli-12-3" # SOURCE
    }

class ValidationConfig:
    DATA_ARGS = {
        "validation_dir": "gamechangerml/data/validation",
        "evaluation_dir": "gamechangerml/data/evaluation",
        "squad": {
            "dev": "squad2.0/dev-v2.0.json",
            "train": "squad2.0/train-v2.0.json",
            "sample_limit": 10
        },
        "nli": {
            "matched": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
            "mismatched": "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
            "train": "multinli_1.0/multinli_1.0_train.jsonl",
        },
        "msmarco": {
            "collection": "msmarco_1k/collection.json",
            "queries": "msmarco_1k/queries.json",
            "relations": "msmarco_1k/relations.json",
            "metadata": "msmarco_1k/metadata.json",
        }
    }
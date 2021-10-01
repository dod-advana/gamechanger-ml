from datetime import datetime
from os import environ
import os
from gamechangerml import REPO_PATH


class DefaultConfig:

    DATA_DIR = os.path.join(REPO_PATH, "common/data/processed")
    LOCAL_MODEL_DIR = os.path.join(REPO_PATH, "gamechangerml/models")
    DEFAULT_FILE_PREFIX = datetime.now().strftime("%Y%m%d")


class S3Config:
    STORE_S3 = True
    S3_MODELS_DIR = "models/v3/"
    S3_CORPUS_DIR = "corpus/"


class D2VConfig:
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
            "model": "bert-base-uncased",
            # If you have a pre-trained model, you can add the model class here.
            "custom_model": None,
            # If you have a custom tokenizer, you can add the tokenizer here.
            "custom_tokenizer": None,
            # Needs to be negative, but allows you to pick which layer you want the embeddings to come from.
            "hidden": -2,
            # It can be 'mean', 'median', or 'max'. This reduces the embedding layer for pooling.
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


class QAConfig:
    MODEL_ARGS = {
        "model_name": "bert-base-cased-squad2",
        # options are: ['scored_answer', 'simple_answer']
        "qa_type": "scored_answer",
        "nbest": 1,  # number of answers to retrieve from each context for comparison
        # if diff between the answer score and null answer score is greater than this threshold, don't return answer
        "null_threshold": -3,
    }


class EmbedderConfig:
    MODEL_ARGS = {
        "model_name": "msmarco-distilbert-base-v2",  # SOURCE
        "embeddings": {
            "embeddings": "embeddings.npy",
            "dataframe": "data.csv",
            "ids": "doc_ids.txt",
        },
        "encoder": {  # args for making the embeddings index
            "min_token_len": 10,
            "overwrite": False,
            "verbose": True,  # for creating LocalCorpus
            "return_id": True,  # for creating LocalCorpus
        },
        "retriever": {"n_returns": 5},  # args for retrieving the vectors
        "retriever": {"n_returns": 5},  # args for retrieving the vectors
        "finetune": {
            "shuffle": True,
            "batch_size": 16,
            "epochs": 1,
            "warmup_steps": 100,
        },
    }


class SimilarityConfig:
    MODEL_ARGS = {"model_name": "distilbart-mnli-12-3"}  # SOURCE


class WordSimConfig:
    MODEL_ARGS = {
        "model_name": "gamechangerml/models/wiki-news-300d-1M.vec"}  # SOURCE


class QEConfig:
    MODEL_ARGS = {
        "init": {  # args for creating QE object
            "qe_model_dir": "gamechangerml/models/qexp_20211001",
            "qe_files_dir": "gamechangerml/src/search/query_expansion",
            "method": "emb",
            "vocab_file": "word-freq-corpus-20201101.txt",
        },
        "expansion": {  # configs for getting expanded terms
            "topn": 2,
            "threshold": 0.2,
            "min_tokens": 3,
        },
    }


class ValidationConfig:
    DATA_ARGS = {
        # need to have validation data in here
        "validation_dir": "gamechangerml/data/validation",
        "evaluation_dir": "gamechangerml/data/evaluation",
        # location with smaller set of corpus JSONs
        "test_corpus_dir": "gamechanger/data/test_corpus",
        "squad": {"dev": "squad2.0/dev-v2.0.json", "train": "squad2.0/train-v2.0.json"},
        "nli": {
            "matched": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
            "mismatched": "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
        },
        "msmarco": {
            "collection": "msmarco_1k/collection.json",
            "queries": "msmarco_1k/queries.json",
            "relations": "msmarco_1k/relations.json",
            "metadata": "msmarco_1k/metadata.json",
        },
        "question_gc": {"queries": "QA_domain_data.json"},
        "retriever_gc": {"gold_standard": "gold_standard.csv"},
        "matamo_feedback_file": "matamo_feedback.csv",
        "search_history_file": "SearchPdfMapping.csv",
        "qe_gc": "QE_domain.json",
    }


class TrainingConfig:
    DATA_ARGS = {
        "training_data_dir": "gamechangerml/data/training",
        "train_test_split_ratio": 0.8,
    }

import requests
import logging
import pytest
import os
import json
import time
import pandas as pd
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.src.utilities.test_utils import open_json, get_most_recent_dir, delete_files

logger = logging.getLogger()
model_path_dict = get_model_paths()
training_dir= "gamechangerml/data/test"
http = requests.Session()

SENT_INDEX = model_path_dict['sentence']
GC_ML_HOST = os.environ.get("GC_ML_HOST", default="localhost")
API_URL = f"{GC_ML_HOST}:5000" if "http" in GC_ML_HOST else f"http://{GC_ML_HOST}:5000"
ENCODER = "multi-qa-MiniLM-L6-cos-v1"


@pytest.mark.order(1)
def test_call_finetune():
    model_dict = {
        "build_type": "sent_finetune",
        "model": ENCODER,
        "batch_size": 8,
        "epochs": 1,
        "warmup_steps": 100,
        "remake_train_data": True,
        "testing_only": True
        }
    resp = http.post(API_URL + "/trainModel", json=model_dict)
    assert resp.ok == True

@pytest.mark.order(2)
def test_made_eval_data():

    ## test created eval data
    test_any = json.loads(open_json("intelligent_search_data.json", "gamechangerml/data/test_data/test_validation/any"))
    test_silver = json.loads(open_json("intelligent_search_data.json", "gamechangerml/data/test_data/test_validation/silver"))
    test_gold = json.loads(open_json("intelligent_search_data.json", "gamechangerml/data/test_data/test_validation/gold"))

    eval_path = get_most_recent_dir("gamechangerml/data/validation/domain/sent_transformer")
    logger.info(f"Looking for updated eval data in {eval_path}")

    for i in range(60):
        logger.info(f"Waited {str(i*5)} seconds for validation data...")
        if os.path.isfile(os.path.join(eval_path, "gold", "intelligent_search_data.json")):
            break
        else:
            time.sleep(5)
    
    gold = json.loads(open_json("intelligent_search_data.json", os.path.join(eval_path, "gold")))
    silver = json.loads(open_json("intelligent_search_data.json", os.path.join(eval_path, "silver")))
    any_ = json.loads(open_json("intelligent_search_data.json", os.path.join(eval_path, "any")))
        
    assert gold['correct_vals'] == test_gold['correct_vals']
    assert gold['incorrect_vals'] == test_gold['incorrect_vals']
    assert silver['correct_vals'] == test_silver['correct_vals']
    assert silver['incorrect_vals'] == test_silver['incorrect_vals']
    assert any_['correct_vals'] == test_any['correct_vals']
    assert any_['incorrect_vals'] == test_any['incorrect_vals']

@pytest.mark.order(3)
def test_made_training_data():

    ## test training data
    training_path = get_most_recent_dir("gamechangerml/data/training/sent_transformer")
    logger.info(f"Looking for training data in {training_path}")

    for i in range(60):
        logger.info(f"Waited {str(i*5)} seconds for training data...")
        if os.path.isfile(os.path.join(training_path, "training_metadata.json")):
            break
        else:
            time.sleep(5)
    
    meta = open_json("training_metadata.json", training_path)
    df = pd.read_csv(os.path.join(training_path, "retrieved_paragraphs.csv"))

    ## checking query - result pairs
    df = pd.read_csv(os.path.join(training_path, "retrieved_paragraphs.csv"))
    assert df['num_matching_docs'].min() >= 1 # must have at least 1 matching doc
    assert df['num_nonmatching_docs'].min() >= 1 # must have at least 1 nonmatching doc
    assert df['overlap'].sum() == 0 # should not be overlap between matching/nonmatching
    assert df['par_balance'].min() >= 0.2 # check not too many negative samples

    ## checking metadata
    assert meta['n_queries'] == '21 train queries / 5 test queries'
    assert int(meta['total_train_samples_size']) > 21
    assert int(meta['total_test_samples_size']) > 5
    assert int(meta['not_found_search_pairs']) == 0

@pytest.mark.order(4)
def test_finetuned_model():

    ## wait for model files
    model_path = os.path.join("gamechangerml", "models", "transformers", ENCODER + "_TEST")
    for i in range(400):
        logger.info(f"Waited {str(i*5)} seconds for finetuned model...")
        if os.path.isdir(model_path):
            if os.path.isfile(os.path.join(model_path, "metadata.json")):
                time.sleep(2)
                break
        else:
            time.sleep(5)

    meta = open_json("metadata.json", model_path)
    assert meta["n_training_samples"] > 100

@pytest.mark.order(5)
def test_made_evals():

    model_path = os.path.join("gamechangerml", "models", "transformers", ENCODER + "_TEST")
    ## wait for evals
    gold_evals_path = os.path.join(model_path, "evals_gc", "gold")
    for i in range(400):
        logger.info(f"Waited {str(i*5)} seconds for evaluation...")
        if os.path.isdir(gold_evals_path):
            if len(os.listdir(gold_evals_path)) > 0:
                break
        else:
            time.sleep(5)

    eval_file = [i for i in os.listdir(gold_evals_path) if "retriever_eval" in i][0]
    gold_evals = open_json(eval_file, gold_evals_path)
    assert gold_evals["query_count"] == 33
    assert gold_evals["MRR"] > 0
    assert gold_evals["mAP"] > 0
    assert gold_evals["recall"] > 0

@pytest.mark.order(6)
def test_cleanup_files():

    eval_path = get_most_recent_dir("gamechangerml/data/validation/domain/sent_transformer")
    training_path = get_most_recent_dir("gamechangerml/data/training/sent_transformer")
    model_path = os.path.join("gamechangerml", "models", "transformers", ENCODER + "_TEST")

    ## clean up files
    delete_files(training_path)
    delete_files(eval_path)
    delete_files(model_path)

    assert not os.path.isdir(eval_path)
    assert not os.path.isdir(training_path)
    assert not os.path.isdir(model_path)

import requests
import logging
import os
import json
import time
import pandas as pd
import pytest
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

@pytest.mark.run(order=1)
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

@pytest.mark.run(order=2)
def test_made_validation_data():
    eval_path = get_most_recent_dir("gamechangerml/data/validation/domain/sent_transformer")
    logger.info(f"**** Looking for updated eval data in {eval_path}")
    for i in range(180):
        if os.path.isfile(os.path.join(eval_path, "gold", "intelligent_search_data.json")):
            break
        else:
            logger.info(f"Countdown: {str(180 - (i * 5))} seconds left...")
            time.sleep(5)

    assert os.path.isfile(os.path.join(eval_path, "gold", "intelligent_search_data.json"))

    ## check created validation data
    test_any = json.loads(open_json("intelligent_search_data.json", "gamechangerml/data/test_data/test_validation/any"))
    test_silver = json.loads(open_json("intelligent_search_data.json", "gamechangerml/data/test_data/test_validation/silver"))
    test_gold = json.loads(open_json("intelligent_search_data.json", "gamechangerml/data/test_data/test_validation/gold"))

    eval_path = get_most_recent_dir("gamechangerml/data/validation/domain/sent_transformer")
    gold = json.loads(open_json("intelligent_search_data.json", os.path.join(eval_path, "gold")))
    silver = json.loads(open_json("intelligent_search_data.json", os.path.join(eval_path, "silver")))
    any_ = json.loads(open_json("intelligent_search_data.json", os.path.join(eval_path, "any")))
    
    assert gold['correct_vals'] == test_gold['correct_vals']
    assert gold['incorrect_vals'] == test_gold['incorrect_vals']
    assert silver['correct_vals'] == test_silver['correct_vals']
    assert silver['incorrect_vals'] == test_silver['incorrect_vals']
    assert any_['correct_vals'] == test_any['correct_vals']
    assert any_['incorrect_vals'] == test_any['incorrect_vals']

@pytest.mark.run(order=3)
def test_made_training_data():
    
    training_path = get_most_recent_dir("gamechangerml/data/training/sent_transformer")
    logger.info(f"**** Looking for training data in {training_path}")
    for i in range(180):
        if os.path.isfile(os.path.join(training_path, "training_metadata.json")):
            break
        else:
            logger.info(f"Countdown: {str(180 - (i * 5))} seconds left...")
            time.sleep(5)

    assert os.path.isfile(os.path.join(training_path, "training_metadata.json"))

    df = pd.read_csv(os.path.join(training_path, "retrieved_paragraphs.csv"))
    assert df['num_matching_docs'].min() >= 1 # must have at least 1 matching doc
    assert df['num_nonmatching_docs'].min() >= 1 # must have at least 1 nonmatching doc
    assert df['overlap'].sum() == 0 # should not be overlap between matching/nonmatching
    assert df['par_balance'].min() >= 0.2 # check not too many negative samples

    meta_td = open_json("training_metadata.json", training_path)
    assert meta_td['n_queries'] == '21 train queries / 5 test queries'
    assert int(meta_td['total_train_samples_size']) > 21
    assert int(meta_td['total_test_samples_size']) > 5
    assert int(meta_td['not_found_search_pairs']) == 0

@pytest.mark.run(order=4)
def test_finetuned_model():
    model_path = os.path.join("gamechangerml", "models", "transformers", ENCODER + "_TEST")
    logger.info(f"Looking for finetuned model in {model_path}")
    for i in range(180):
        if os.path.isfile(os.path.join(model_path, "metadata.json")):
            break
        else:
            logger.info(f"Countdown: {str(180 - (i * 5))} seconds left...")
            time.sleep(5)

    assert os.path.isfile(os.path.join(model_path, "metadata.json"))

    meta_mod = open_json("metadata.json", model_path)
    assert meta_mod["n_training_samples"] > 100

@pytest.mark.run(order=5)
def test_evaluated_model():
    model_path = os.path.join("gamechangerml", "models", "transformers", ENCODER + "_TEST")
    gold_evals_path = os.path.join(model_path, "evals_gc", "gold")
    logger.info(f"Looking for model evals in {gold_evals_path}")
    for i in range(180):
        if os.path.isdir(gold_evals_path):
            if len(os.listdir(gold_evals_path)) > 0:
                break
            else:
                logger.info(f"Countdown: {str(180 - (i * 5))} seconds left...")
                time.sleep(5)
        else:
            logger.info(f"Countdown: {str(180 - (i * 5))} seconds left...")
            time.sleep(5)

    ## check created evals
    eval_file = [i for i in os.listdir(gold_evals_path) if "retriever_eval" in i][0]
    gold_evals = open_json(eval_file, gold_evals_path)
    assert gold_evals["query_count"] == 33
    assert gold_evals["MRR"] > 0
    assert gold_evals["mAP"] > 0
    assert gold_evals["recall"] > 0

@pytest.mark.run(order=6)
def test_delete_files():

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

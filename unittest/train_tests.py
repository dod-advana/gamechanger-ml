import pytest
import requests
import json
import time

#from gamechangerml.api.fastapi.routers.controls import train_model

API_URL = "http://localhost:5000/trainModel"
CORPUS_DIR = 'gamechangerml/corpus'
#TRAINING_DATA

false = False
expected = {
        "process_status": {
            "flags": {
                "corpus: corpus_download": false,
                "training: train_model": false,
                "training: load_corpus": false,
                "models: reloading_models": false
            }
        },
        "completed_process": []
        }

## API tests
async def send_train(model_dict):
    '''Send test functions to API'''
    
    data = json.dumps(model_dict).encode("utf-8")
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, data = data, headers = headers)

    return response.json()

@pytest.mark.asyncio
async def test_finetune():
    
    model_dict = {
        "build_type": "sent_finetune",
        "batch_size": 32,
        "epochs": 1,
        "warmup_steps": 100
    }
    resp = await send_train(model_dict)
    
    print(f"\nResponse: {resp}")
    assert resp == expected

@pytest.mark.asyncio
async def test_build_sent():

    model_dict = {
        "build_type": "sentence",
        "corpus": CORPUS_DIR,
        "encoder_model": 'msmarco-bert-base-cased-v2',
        "gpu": False,
        "upload": False,
        "version": 5,
    }
    resp = await send_train(model_dict)
    
    print(f"\nResponse: {resp}")
    assert resp == expected

@pytest.mark.asyncio
async def test_build_qexp():

    model_dict = {
        "build_type": "qexp",
        "model_id": '20211101',
        "validate": False,
        "upload": False,
        "version": 4,
    }
    resp = await send_train(model_dict)
    
    print(f"\nResponse: {resp}")
    assert resp == expected

@pytest.mark.asyncio
async def test_eval():

    models = ['msmarco-bert-base-cased-v2', 'bert-base-cased-squad2', 'distilbart-mnli-12-3', 'sent_index_20210715']
    for i in models:
        model_dict = {
            "build_type": "eval",
            "model_name": i,
            "skip_original": False,
            "sample_limit": 10,
            "validation_data": "latest"
        }
        resp = await send_train(model_dict)
    
        print(f"\nResponse: {resp}")
        assert resp == expected

## non-API tests


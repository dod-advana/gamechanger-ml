import pytest
import requests
import json
import time

#from gamechangerml.api.fastapi.routers.controls import train_model

API_URL = "http://localhost:5000/trainModel"

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


#def test_build_qexp():


#def test_eval():
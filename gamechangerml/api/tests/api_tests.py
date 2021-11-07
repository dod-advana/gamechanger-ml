import requests
import logging
import pytest
import os
import json
import sys
import time

from gamechangerml.src.search.query_expansion.utils import remove_original_kw
from .test_examples import TestSet

logger = logging.getLogger()
GC_ML_HOST = os.environ.get("GC_ML_HOST", default="localhost")
API_URL = f"http://{GC_ML_HOST}:5000"
QA_TIMEOUT = 20


def test_conn():
    resp = requests.get(API_URL)
    assert resp.ok == True


def test_expandTerms():
    test_data = {"termsList": ["artificial intelligence"]}
    resp = requests.post(API_URL + "/expandTerms", json=test_data)
    verified = {"artificial intelligence": [
        '"intelligence"', '"human intelligence"']}
    assert resp.json() == verified


# this is in here because it is based off of api function flow not specifically qe
def test_remove_kw_1():
    test_term = "network"
    test_list = ["network connection", "communications network"]
    terms = remove_original_kw(test_list, test_term)
    verified = ["connection", "communications"]
    assert terms == verified


def test_remove_kw_2():
    test_term = "animal"
    test_list = ["animals", "animal cruelty"]
    terms = remove_original_kw(test_list, test_term)
    verified = ["animals", "cruelty"]
    assert terms == verified


def test_remove_kw_3():
    test_term = "american navy"
    test_list = ["british navy", "navy washington"]
    terms = remove_original_kw(test_list, test_term)
    verified = ["british navy", "navy washington"]
    assert terms == verified


def test_remove_kw_4():
    test_term = "weapons"
    test_list = ["enemy weapons", "weapons explosives"]
    terms = remove_original_kw(test_list, test_term)
    verified = ["enemy", "explosives"]
    assert terms == verified


def test_getTransformerList():
    resp = requests.get(API_URL + "/getTransformerList")
    verified = TestSet.transformer_list_expect
    assert resp.json() == verified
    return verified


def getCurrentTrans():
    resp = requests.get(API_URL + "/getCurrentTransformer")
    return resp.json()


def test_changeModels():

    test_transformer = "distilroberta-base"
    model_dict = {"model_name": test_transformer}
    resp = requests.post(API_URL + "/updateModel", json=model_dict)
    time.sleep(10)
    curr = getCurrentTrans()
    assert curr["model_name"] == resp.json()["model_name"]

## Train Model tests

def test_trainModel_sentence():
    model_dict = {
        "build_type": "sentence",
        "corpus": "gamechangerml/data/test_data", # should have 3 test docs
        "encoder_model": "msmarco-distilbert-base-v2",
        "gpu": False,
        "upload": False,
        "version": "TEST"
    }
    resp = requests.post(API_URL + "/trainModel", json=model_dict)
    assert resp.ok == True
    checks = 0
    while checks < 15:
        time.sleep(15)
        checks += 1
        if len(resp.json()["completed_process"]) < 2:
            print("Time: {}. Completed response: {}".format(checks * 15, resp.json()["completed_process"]))
            break
    assert resp.json()["completed_process"] == ["load_corpus", "train_model"]

def test_trainModel_sent_finetune():
    model_dict = {
        "build_type": "sent_finetune",
        "batch_size": 32,
        "epochs": 1,
        "warmup_steps": 100,
        "testing_only": True
    }
    resp = requests.post(API_URL + "/trainModel", json=model_dict)
    assert resp.ok == True
    checks = 0
    while checks < 15:
        time.sleep(15)
        checks += 1
        if len(resp.json()["completed_process"]) < 1:
            print("Time: {}. Completed response: {}".format(checks * 15, resp.json()["completed_process"]))
            break
    assert resp.json()["completed_process"] == ["train_model"]

#def test_trainModel_eval():

    ## squad

    ## msmarco

    ## nli

    ## qexp

## Search Tests

def test_transformerSearch():
    test_data = TestSet.transformer_test_data
    verified = TestSet.transformer_search_expect

    resp = requests.post(API_URL + "/transformerSearch", json=test_data)
    assert resp.json() == verified

def test_transformerSearch():
    test_data = TestSet.qa_test_data
    verified = TestSet.qa_expect

    resp = requests.post(API_URL + "/questionAnswer", json=test_data)
    assert resp.json() == verified

def test_postSentSearch():
    test_data = TestSet.sentence_test_data
    verified = TestSet.sentence_search_expect

    resp = requests.post(API_URL + "/transSentenceSearch", json=test_data)

    assert resp.json() == verified
    return verified

## QA Tests

def send_qa(query, context):
    
    start = time.perf_counter()
    post = {
        "query": query,
        "search_context": context
    }
    data = json.dumps(post).encode("utf-8")
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL + "/questionAnswer", data = data, headers = headers)

    end = time.perf_counter()
    took = float(f"{end-start:0.4f}")

    return response.json(), took
    
qa_test_context_1 = [
        "Virginia'\''s Democratic-controlled Legislature passed a bill legalizing the possession of small amounts of marijuana on Wednesday, making it the 16th state to take the step. Under Virginia'\''s new law, adults ages 21 and over can possess an ounce or less of marijuana beginning on July 1, rather than Jan. 1, 2024. Gov. Ralph Northam, a Democrat, proposed moving up the date, arguing it would be a mistake to continue to penalize people for possessing a drug that would soon be legal. Lt. Gov. Justin Fairfax, also a Democrat, broke a 20-20 vote tie in Virginia'\''s Senate to pass the bill. No Republicans supported the measure. Democratic House of Delegates Speaker Eileen Filler-Corn hailed the plan. Today, with the Governor'\''s amendments, we will have made tremendous progress in ending the targeting of Black and brown Virginians through selective enforcement of marijuana prohibition by this summer she said in a statement. Republicans voiced a number of objections to what they characterized as an unwieldy, nearly 300-page bill. Several criticized measures that would grant licensing preferences to people and groups who'\''ve been affected by the war on drugs and make it easier for workers in the industry to unionize. Senate Minority Leader Tommy Norment also questioned Northam'\''s motives.",
        "We have a governor who wants to contribute to the resurrection of his legacy, Norment said, referring to the 2019 discovery of a racist photo in Northam'\''s 1984 medical school yearbook. The accelerated timeline sets Virginia cannabis consumers in an unusual predicament. While it will be legal to grow up to four marijuana plants beginning July 1, it could be several years before the state begins licensing recreational marijuana retailers. And unlike other states, the law won'\''t allow the commonwealth'\''s existing medical dispensaries to begin selling to all adults immediately. Jenn Michelle Pedini, executive director of Virginia NORML, called legalization an incredible victory but said the group would continue to push to allow retail sales to begin sooner.",
        "In the interest of public and consumer safety, Virginians 21 and older should be able to purchase retail cannabis products at the already operational dispensaries in 2021, not in 2024, Pedini said in a statement. Such a delay will only exacerbate the divide for equity applicants and embolden illicit activity. Northam and other Democrats pitched marijuana legalization as a way to address the historic harms of the war on drugs. One state study found Black Virginians were 3.5 times more likely to be arrested on marijuana charges compared with white people. Those trends persisted even after Virginia reduced penalties for possession to a $25 civil fine. New York and New Jersey also focused on addressing those patterns when governors in those states signed laws to legalize recreational marijuana this year. Northam'\''s proposal sets aside 30% of funds to go to communities affected by the war on drugs, compared with 70% in New Jersey. Another 40% of Virginia'\''s revenue will go toward early childhood education, with the remainder funding public health programs and substance abuse treatment.",
        "Those plans, and much of the bill'\''s regulatory framework, are still tentative; Virginia lawmakers will have to approve them again during their general session next year. Some criminal justice advocates say lawmakers should also revisit language that creates a penalty for driving with an open container of marijuana. In the absence of retail sales, some members of law enforcement said it'\''s not clear what a container of marijuana will be. The bill specifies a category of social equity applicants, such as people who'\''ve been charged with marijuana-related offenses or who graduated from historically Black colleges and universities. Those entrepreneurs will be given preference when the state grants licensing. Mike Thomas, a Black hemp cultivator based in Richmond who served jail time for marijuana possession, said those entrepreneurs deserved special attention. Thomas said he looked forward to offering his own line of organic, craft cannabis. Being that the arrest rate wasn'\''t the same for everyone, I don'\''t think the business opportunities should be the same for everyone"
    ]
    
def test_qa_regular():
    query = "when is marijuana legalized"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_qa_one_question():
    query = "when is marijuana legalized?"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_qa_multiple_question():
    query = "when is marijuana legalized???"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_qa_allcaps():
    query = "WHEN IS MARIJUANA LEGALIZED"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_qa_apostrophe():
    query = "when's marijuana legalized"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_qa_past_tense():
    query = "when was marijuana legalized?"
    expected = 'Wednesday'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_qa_future_tense():
    query = "when will marijuana be legal?"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_qa_specific():
    query = "when will marijuana be legal in Virginia?"
    expected = 'it will be legal to grow up to four marijuana plants beginning July 1'
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer

def test_qa_outside_scope():
    query = "what is the capital of Assyria?"
    expected = ''
    resp, took = send_qa(query, qa_test_context_1)
    top_answer = resp['answers'][0]['text']
    scores = [i['null_score_diff'] for i in resp['answers']]
    print("\nQUESTION: ", query, "\nANSWER: ", top_answer, f"\n (took {took} seconds)\n")
    assert top_answer == expected # assert response is right
    assert took < QA_TIMEOUT # assert time
    assert resp['answers'][0]['null_score_diff'] == min(scores) # assert is best scoring answer
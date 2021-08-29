from sentence_transformers import SentenceTransformer, InputExample, util, losses
from torch.utils.data import DataLoader
import pandas as pd
import os
import random

from gamechangerml.src.model_testing.validation_data import IntelSearchData
from gamechangerml.src.utilities.es_search_utils import get_paragraph_results, connect_es
from gamechangerml.src.utilities.model_helper import timestamp_filename
from gamechangerml.configs.config import ValidationConfig, EmbedderConfig
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.api.utils.logger import logger

model_path_dict = get_model_paths()

VALIDATION_DIR = ValidationConfig.DATA_ARGS['validation_directory']
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
BASE_MODEL_NAME = EmbedderConfig.MODEL_ARGS['model_name']
training_data_csv_path = os.path.join(VALIDATION_DIR, timestamp_filename('finetune_sent_data', '.csv'))
model_load_path = os.path.join(LOCAL_TRANSFORMERS_DIR, BASE_MODEL_NAME)
model_save_path = os.path.join(LOCAL_TRANSFORMERS_DIR, timestamp_filename(BASE_MODEL_NAME + '_finetuned', '/'))

## load original model
model = SentenceTransformer(model_load_path)

random.seed(42)

## make data
def collect_training_data():

    es = connect_es()
    intel = IntelSearchData()
    correct, incorrect = intel.filter_rels(min_correct_matches=2)

    query_results = []
    for i in correct.keys():
        query = intel.queries[i]
        for k in correct[i]:
            doc = intel.collection[k]
            uid = str(i + '_' + k)
            try:
                para = get_paragraph_results(es, query, doc)[0][0]
            except:
                para = None
            texts = [query, para]
            query_results.append([query, doc, uid, para, texts, 0.95])

    for i in incorrect.keys():
        query = intel.queries[i]
        for k in incorrect[i]:
            doc = intel.collection[k]
            uid = str(i + '_' + k)
            try:
                para = get_paragraph_results(es, query, doc)[0][0]
            except:
                para = None
            texts = [query, para]
            query_results.append([query, doc, uid, para, texts, 0.05])

    df = pd.DataFrame(query_results, columns=['query', 'doc', 'uid', 'paragraph', 'texts', 'score'])

    df.to_csv(training_data_csv_path)

    return df.dropna(subset = ['paragraph']).reset_index()

def split_train_test(df):
    '''Split df into train/test set'''

    train = df.sample(frac=0.8, replace=False, random_state=42).reset_index()
    test = df[~df.index.isin(train.index)].reset_index()
    assert(train.shape[0] + test.shape[0] == df.shape[0])
    logger.info("Number of training samples: ", train.shape[0])
    logger.info("Number of test samples: ", test.shape[0])

    return train, test

def get_cos_sim(model, pair):
    emb1 = model.encode(pair[0])
    emb2 = model.encode(pair[1])
    cos_sim = util.cos_sim(emb1, emb2)
    
    return cos_sim

def finetune_sent_transformer(train, test, model):
    '''Prep finetune samples, get cos_sim scores before finetuning'''

    ## make train samples for finetuning
    train_samples = []
    train_cos_sim = []
    for i in train.index:
        uid = train.loc[i, 'uid']
        texts = train.loc[i, 'texts']
        score = train.loc[i, 'score']
        inputex = InputExample(uid, texts, score)
        train_samples.append(inputex)
        cos_sim = get_cos_sim(model, texts)
        train_cos_sim.append(cos_sim)
    
    train['orig_cos_sim'] = train_cos_sim

    test_cos_sim = []
    for i in test.index:
        texts = test.loc[i, 'texts']
        cos_sim = get_cos_sim(model, texts)
        test_cos_sim.append(cos_sim)
    
    test['orig_cos_sim'] = test_cos_sim

    return train, test, train_samples

def finetune_model(model, train_samples, train, test, model_save_path):
    '''Finetune the model and get new cos sim scores'''

    ## finetune on samples
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

    ## save model
    model.save(model_save_path)

    ## get new cos_sim scores
    train_cos_sim = []
    for i in train.index:
        texts = train.loc[i, 'texts']
        cos_sim = get_cos_sim(model, texts)
        train_cos_sim.append(cos_sim)
    
    train['new_cos_sim'] = train_cos_sim

    test_cos_sim = []
    for i in test.index:
        texts = test.loc[i, 'texts']
        cos_sim = get_cos_sim(model, texts)
        test_cos_sim.append(cos_sim)
    
    test['new_cos_sim'] = test_cos_sim

    return train, test

def compare_cos_sim(df):

    df['difference'] = df['new_cos_sim'] - df['orig_cos_sim']

    return df
    
def __main__():

    df = collect_training_data()
    train, test = split_train_test(df)
    train, test, samples = finetune_sent_transformer(train, test, model=model)
    train, test = finetune_model(samples, train, test, model=model, model_save_path=model_save_path)

    ## print stats
    ## test working
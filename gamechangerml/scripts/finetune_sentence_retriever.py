from sentence_transformers import SentenceTransformer, InputExample, util, losses
from torch.utils.data import DataLoader
import pandas as pd
import os
import random
import argparse

from gamechangerml.src.model_testing.validation_data import IntelSearchData
from gamechangerml.src.utilities.es_search_utils import get_paragraph_results, connect_es
from gamechangerml.src.utilities.model_helper import timestamp_filename
from gamechangerml.configs.config import ValidationConfig, EmbedderConfig
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.api.utils.logger import logger

model_path_dict = get_model_paths()

ES_URL = 'https://vpc-gamechanger-iquxkyq2dobz4antllp35g2vby.us-east-1.es.amazonaws.com'
VALIDATION_DIR = ValidationConfig.DATA_ARGS['validation_dir']
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
BASE_MODEL_NAME = EmbedderConfig.MODEL_ARGS['model_name']

random.seed(42)

def _collect_results(query_results, subset_dict, intel, es, score):
    '''Format queries'''

    for i in subset_dict.keys():
        query = intel.queries[i]
        for k in subset_dict[i]:
            doc = intel.collection[k]
            uid = str(i + '_' + k)
            try:
                para = get_paragraph_results(es, query, doc)[0][0]
                para = ' '.join(para.split(' ')[:400]) # truncate to 400 tokens
            except:
                para = None
            texts = [query, para]
            query_results.append([query, doc, uid, para, texts, score])

    return query_results

def _get_cos_sim(model, pairs, colname, df):

    sim_scores = []
    for i in pairs:
        emb1 = model.encode(i[0])
        emb2 = model.encode(i[1])
        cos_sim = float(util.cos_sim(emb1, emb2))
        sim_scores.append(cos_sim)
    
    df[colname] = sim_scores
    
    return df

def _compare_cos_sim(df, name):

    df['difference'] = df['new_cos_sim'] - df['orig_cos_sim']
    df['difference'] = df['difference'].apply(lambda x: abs(x))

    logger.info("Median cosine similarity change for {} after fine-tuning: {}".format(name, df['difference'].median()))

    return df

def collect_training_data():
    '''Get paragraphs from ES for query/doc matches; save to csv'''

    es = connect_es(ES_URL)
    intel = IntelSearchData()
    correct, incorrect = intel.filter_rels(min_correct_matches=2)

    query_results = _collect_results(query_results=[], subset_dict=correct, intel=intel, es=es, score=0.95)
    query_results = _collect_results(query_results=query_results, subset_dict=incorrect, intel=intel, es=es, score=0.05)

    df = pd.DataFrame(query_results, columns=['query', 'doc', 'uid', 'paragraph', 'texts', 'score'])
    df = df.dropna(subset = ['paragraph']).reset_index() # drop rows with no paragraph

    df.to_csv(training_data_csv_path)
    logger.info("Training data saved to {}".format(str(training_data_csv_path)))

    return df

def split_train_test(df, split_ratio):
    '''Split df into train/test set'''

    train = df.sample(frac=split_ratio, replace=False, random_state=42)
    test = df[~df.index.isin(train.index)]
    assert(train.shape[0] + test.shape[0] == df.shape[0])
    logger.info("Number of training samples: {}".format(str(train.shape[0])))
    logger.info("Number of test samples: {}".format(str(test.shape[0])))

    train['type'] = "train"
    test['type'] = "test"

    return train, test

def finetune(train, test, model, model_save_path, shuffle, batch_size, epochs, warmup_steps):
    '''Get before/after cosine similarity and finetune the model'''

    train.reset_index(inplace = True)
    test.reset_index(inplace = True)

    ## make train samples for finetuning
    train_samples = []
    for i in train.index:
        uid = str(i)
        texts = [train.loc[i, 'query'], train.loc[i, 'paragraph']]
        score = float(train.loc[i, 'score'])
        inputex = InputExample(uid, texts, score)
        train_samples.append(inputex)

    train = _get_cos_sim(model, train['texts'], 'orig_cos_sim', train)
    test = _get_cos_sim(model, test['texts'], 'orig_cos_sim', test)

    ## finetune on samples
    train_dataloader = DataLoader(train_samples, shuffle=shuffle, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps)

    ## save model
    model.save(model_save_path)
    logger.info("Finetuned model saved to {}".format(str(model_save_path)))

    train = _get_cos_sim(model, train['texts'], 'new_cos_sim', train)
    test = _get_cos_sim(model, test['texts'], 'new_cos_sim', test)

    train = _compare_cos_sim(train, 'train')
    test = _compare_cos_sim(test, 'test')

    return train, test
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finetuning the sentence transformer model with in-domain data")
    
    parser.add_argument(
        "--data-path", "-d", 
        dest="data_path", 
        help="path to csv with finetuning data"
        )

    parser.add_argument(
        "--model-path", "-m", 
        dest="model_path", 
        help="path to model for fine-tuning"
        )

    args = parser.parse_args()

    
    logger.info("|------------------Collecting training data------------------|")

    if args.data_path:
        training_data_csv_path = args.data_path
        df = pd.read_csv(training_data_csv_path)
    else:
        training_data_csv_path = os.path.join(VALIDATION_DIR, timestamp_filename('finetune_sent_data', '.csv'))
        df = collect_training_data() 

    logger.info("|---------------------Splitting train/test-------------------|")
    split_ratio = EmbedderConfig.MODEL_ARGS['train_proportion']
    train, test = split_train_test(df, split_ratio)

    logger.info("|---------------------Finetuning model-----------------------|")
    if args.model_path:
        model_load_path = args.model_path
    else:
        model_load_path = os.path.join(LOCAL_TRANSFORMERS_DIR, BASE_MODEL_NAME)

    model_save_path = os.path.join(LOCAL_TRANSFORMERS_DIR, timestamp_filename(BASE_MODEL_NAME + '_finetuned', '/'))

    ## load original model
    model = SentenceTransformer(model_load_path)
    train, test = finetune(train, test, model=model, model_save_path=model_save_path, **EmbedderConfig.MODEL_ARGS['finetune'])

    all_data = pd.concat([train, test])
    all_data.to_csv(training_data_csv_path)
    logger.info("Training data saved to {}".format(str(training_data_csv_path)))
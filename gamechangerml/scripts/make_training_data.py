import argparse
import random
import pandas as pd
from gamechangerml.configs.config import TrainingConfig, ValidationConfig, EmbedderConfig, SimilarityConfig
from gamechangerml.src.search.sent_transformer.model import SentenceEncoder, SentenceSearcher
from gamechangerml.src.utilities.es_search_utils import connect_es, collect_results
from gamechangerml.src.utilities.text_utils import normalize_query
from gamechangerml.src.utilities.test_utils import *
from gamechangerml.api.utils.logger import logger
from gamechangerml.api.utils.pathselect import get_model_paths

model_path_dict = get_model_paths()
random.seed(42)

ES_URL = 'https://vpc-gamechanger-iquxkyq2dobz4antllp35g2vby.us-east-1.es.amazonaws.com'
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]

VALIDATION_DIR = get_most_recent_dir(os.path.join(ValidationConfig.DATA_ARGS["validation_dir"], "sent_transformer"))
SENT_INDEX = model_path_dict["sentence"]
base_dir=TrainingConfig.DATA_ARGS["training_data_dir"]
tts_ratio=TrainingConfig.DATA_ARGS["train_test_split_ratio"]
gold_standard_path = os.path.join(
    "gamechangerml/data/user_data", ValidationConfig.DATA_ARGS["retriever_gc"]["gold_standard"]
    )


def train_test_split(data, tts_ratio):
    '''Splits a dictionary into train/test set based on split ratio'''

    train_size = round(len(data) * tts_ratio)
    train_keys = random.sample(data.keys(), train_size)
    test_keys = [i for i in data.keys() if i not in train_keys]

    train = {k: data[k] for k in train_keys}
    test = {k: data[k] for k in test_keys}

    return train, test

def lookup_negative_samples(
    intel, 
    index_path=SENT_INDEX, 
    transformers_path=LOCAL_TRANSFORMERS_DIR, 
    sim_model_name=SimilarityConfig.MODEL_ARGS['model_name'], 
    encoder_model_name=EmbedderConfig.MODEL_ARGS['encoder_model_name'], 
    n_returns=EmbedderConfig.MODEL_ARGS['n_returns'], 
    encoder=None, 
    sim_model=None
    ):

    def normalize_docs(doc):
        return doc.split('.pdf')[0]
    
    ## look up queries against index
    
    retriever = SentenceSearcher(
        index_path=index_path, 
        transformers_path=transformers_path, 
        sim_model_name=sim_model_name, 
        encoder_model_name=encoder_model_name, 
        n_returns=n_returns, 
        encoder=encoder, 
        sim_model=sim_model
        )

    reverse_docs = {v.upper():k for (k, v) in intel['collection'].items()}
    neutral_samples = {}
    for key in intel['correct'].keys():
        query = intel['queries'][key]
        docs = intel['correct'][key]
        doc_names = [intel['collection'][val] for val in docs]
        try:
            doc_texts, doc_ids, doc_scores = retriever.retrieve_topn(query)
            clean_ids = [normalize_docs(i) for i in doc_ids]
            dedup = []
            for d in clean_ids:
                try:
                    if d not in doc_names:
                        dedup.append(d)
                    else:
                        continue
                except:
                    logger.info(f"------Error finding doc_id for {d}")

            dedup = [d.upper() for d in clean_ids if d not in doc_names]
            diff = len(clean_ids) - len(dedup)
            if diff > 0:
                logger.info(f"Removed {str(diff)} (correct) duplicates")
            neutral_samples[key] = [reverse_docs[d] for d in dedup]
            logger.info(f"Found {str(len(dedup))} negative samples for {key}")
        except:
            logger.info(f"------Error retrieving sent index results for {query}")

    intel['neutral'] = neutral_samples

    return intel

def add_gold_standard(intel, gold_standard_path):
    '''Adds original gold standard data to the intel training data.'''
    gold = pd.read_csv(gold_standard_path, names=['query', 'document'])
    gold['query_clean'] = gold['query'].apply(lambda x: normalize_query(x))
    gold['docs_split'] = gold['document'].apply(lambda x: x.split(';'))
    all_docs = list(set([a for b in gold['docs_split'].tolist() for a in b]))

    def add_key(mydict):
        '''Adds new key to queries/collections dictionaries'''
        last_key = sorted([*mydict.keys()])[-1]
        key_len = len(last_key) - 1
        last_prefix = last_key[0]
        last_num = int(last_key[1:])
        new_num = str(last_num + 1)
        
        return last_prefix + str(str(0)*(key_len - len(new_num)) + new_num)

    # check if queries already in dict, if not add
    for i in gold['query_clean']:
        if i in intel['queries'].values():
            logger.info(f"'{i}' already in intel queries")
            continue
        else:
            logger.info(f"adding '{i}' to intel queries")
            new_key = add_key(intel['queries'])
            intel['queries'][new_key] = i
    
    # check if docs already in dict, if not add
    for i in all_docs:
        if i in intel['collection'].values():
            logger.info(f"'{i}' already in intel collection")
            continue
        else:
            logger.info(f"adding '{i}' to intel collection")
            new_key = add_key(intel['collection'])
            intel['collection'][new_key] = i

    # check if rels already in intel, if not add
    reverse_q = {v:k for k,v in intel['queries'].items()}
    reverse_d = {v:k for k,v in intel['collection'].items()}
    for i in gold.index:
        q = gold.loc[i, 'query_clean']
        docs = gold.loc[i, 'docs_split']
        for j in docs:
            q_id = reverse_q[q]
            d_id = reverse_d[j]
            if q_id in intel['correct']: # if query in rels, add new docs
                if d_id in intel['correct'][q_id]:
                    continue
                else:
                    intel['correct'][q_id] += [d_id]
            else:
                intel['correct'][q_id] = [d_id]
    
    return intel

def make_training_data(base_dir, tts_ratio, gold_standard_path):

    ## open json files
    directory = os.path.join(VALIDATION_DIR, 'any')
    f = open_json('intelligent_search_data.json', directory)
    intel = json.loads(f)

    ## add gold standard samples
    intel = add_gold_standard(intel, gold_standard_path)

    ## collect negative samples
    intel = lookup_negative_samples(intel)
    
    ## set up save dir
    sub_dir = os.path.join(base_dir, 'sent_transformer')
    save_dir = make_timestamp_directory(sub_dir)

    ## query ES
    es = connect_es(ES_URL)
    correct_found, correct_notfound = collect_results(relations=intel['correct'], queries=intel['queries'], collection=intel['collection'], es=es, label=1)
    logger.info(f"---Number of correct query/result pairs that were not found in ES: {str(len(correct_notfound))}")
    neutral_found, neutral_notfound = collect_results(relations=intel['neutral'], queries=intel['queries'], collection=intel['collection'], es=es, label=0)
    logger.info(f"---Number of neutral query/result pairs that were not found in ES: {str(len(neutral_notfound))}")
    incorrect_found, incorrect_notfound = collect_results(relations=intel['incorrect'], queries=intel['queries'], collection=intel['collection'], es=es, label=-1)
    logger.info(f"---Number of incorrect query/result pairs that were not found in ES: {str(len(incorrect_notfound))}")

    ## make sure no dups
    neutral_found = {k:v for (k, v) in neutral_found.items() if k not in correct_found.keys()}
    neutral_notfound = {k:v for (k, v) in neutral_notfound.items() if k not in correct_notfound.keys()}

    ## save a json of the query-doc pairs that did not retrieve an ES paragraph for training data
    notfound = {**correct_notfound, **neutral_notfound, **incorrect_notfound}
    logger.info(f"---Number of total query/result pairs that were not found in ES: {str(len(notfound))}")
    notfound_path = os.path.join(save_dir, 'not_found_search_pairs.json')
    with open(notfound_path, "w") as outfile:
        json.dump(notfound, outfile)

    ## train/test split (separate on correct/incorrect for balance)
    correct_train, correct_test = train_test_split(correct_found, tts_ratio)
    neutral_found_train, neutral_found_test = train_test_split(neutral_found, tts_ratio)
    incorrect_train, incorrect_test = train_test_split(incorrect_found, tts_ratio)
    train = {**neutral_found_train, **correct_train, **incorrect_train}
    test = {**neutral_found_test, **correct_test, **incorrect_test}

    data = {"train": train, "test": test}
    metadata = {
        "date_created": str(date.today()),
        "n_positive_samples": len(correct_found),
        "n_negative_samples": len(incorrect_found),
        "n_neutral_samples": len(neutral_found),
        "train_size": len(train),
        "test_size": len(test),
        "split_ratio": tts_ratio
    }

    ## save data and metadata files
    data_path = os.path.join(save_dir, 'training_data.json')
    metadata_path = os.path.join(save_dir, 'training_metadata.json')

    with open(data_path, "w") as outfile:
        json.dump(data, outfile)

    with open(metadata_path, "w") as outfile:
        json.dump(metadata, outfile)

    return data, metadata

if __name__ == '__main__':

    make_training_data(base_dir=base_dir, tts_ratio=tts_ratio, gold_standard_path=gold_standard_path)
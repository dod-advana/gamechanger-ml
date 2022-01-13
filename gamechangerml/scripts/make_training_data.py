import random
import torch
import pandas as pd
import os
import json
from datetime import date
from typing import List, Union, Dict, Tuple
import spacy


from gamechangerml.configs.config import TrainingConfig, ValidationConfig, SimilarityConfig
from gamechangerml.src.search.sent_transformer.model import SentenceSearcher
from gamechangerml.src.utilities.text_utils import normalize_query
from gamechangerml.src.utilities.test_utils import *
from gamechangerml.api.utils.logger import logger
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.scripts.update_eval_data import make_tiered_eval_data
from gamechangerml import DATA_PATH

model_path_dict = get_model_paths()
random.seed(42)

LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
SIM_MODEL = SimilarityConfig.BASE_MODEL
training_dir= os.path.join(DATA_PATH, "training", "sent_transformer")
tts_ratio=TrainingConfig.DATA_ARGS["train_test_split_ratio"]
gold_standard_path = os.path.join(
    "gamechangerml/data/user_data", ValidationConfig.DATA_ARGS["retriever_gc"]["gold_standard"]
    )

CORPUS_DIR = "gamechangerml/corpus"
corpus_docs = [i.split('.json')[0] for i in os.listdir(CORPUS_DIR) if os.path.isfile(os.path.join(CORPUS_DIR, i))]

def get_sample_paragraphs(pars, par_limit=100, min_length=150):
    '''Collect sample paragraphs longer than min_length (char), up to par_limit paragraphs'''
    
    count = 0
    collected_pars = []
    for i in pars:
        if count < par_limit:
            if len(i['par_raw_text_t']) >= min_length:
                count += 1
                collected_pars.append({"text": i['par_raw_text_t'], "id": i['id']})
        else:
            break
    
    return collected_pars

def get_best_paragraphs(data: pd.DataFrame, query: str, doc_id: str, nlp, min_score: float=0.60) -> List[Dict[str,str]]:
    """Retrieves the best paragraphs for expected doc using similarity model
    Args:
        data [pd.DataFrame]: data df with processed text at paragraph_id level for sent_index
        query [str]: query
        doc_id [str]: doc_id of the expected document to show up with the query
        sim: SimilarityRanker class
        n_matching [int]: number of matching paragraphs to retrieve for the expected doc
    Returns:
        [List[Dict[str,str]]]: List of dictionaries of paragraph matches
    """
    logger.info(f"Retrieving matches for query: {query}, expected doc: {doc_id}")
    pars = []
    doc1 = nlp(query)
    if doc_id not in corpus_docs:
        logger.warning(f"---Did not find {doc_id} in the corpus")

    json = open_json(doc_id + '.json', CORPUS_DIR)
    paragraphs = json['paragraphs']
    sents = get_sample_paragraphs(paragraphs)[:50] # get top 50 paragraphs
    for sent in sents:
        short = ' '.join(sent['text'].split(' ')[:400]) # shorten paragraphs
        pars.append(short)

    ranked = []
    try:
        if len(sents) == 0:
            logger.info("---No paragraphs retrieved for this expected doc")
        elif len(sents) == 1:
            ranked = [{"score": 'na', "id": sents[0]['id'], "text": sents[0]['text']}]
        else:
            comparisons = []
            for sent in sents:
                doc2 = nlp(sent['text'])
                sim = doc1.similarity(doc2)
                if sim >= min_score:
                    record = {"score": sim, "id": sent['id'], "text": sent['text']}
                    comparisons.append(record)
                else:
                    pass
            ranked = sorted(comparisons, key = lambda z: z['score'], reverse=True)
        logger.info(f"*** Collected {str(len(ranked))} / {str(len(sents))} paragraphs (passing sim threshold) retrieved for {doc_id}")
    except Exception as e:
        logger.info(f"---Could not re-rank the paragraphs for {query}")
        logger.warning(e) 
    
    return ranked

def check_no_match(expected_id: str, par_id: str) -> bool:
    """Checks if paragraph ID matches the expected doc ID"""
    if par_id.split('.pdf')[0].upper().strip().lstrip() == expected_id.upper().strip().lstrip():
        return False
    else:
        return True

def get_negative_paragraphs(
    data: pd.DataFrame, query: str, doc_id: str, retriever, n_returns: int) -> List[Dict[str,str]]:
    """Looks up negative (not matching) paragraphs for each query
    Args:
        data [pd.DataFrame]: data df with processed text at paragraph_id level for sent_index
        query [str]: query
        doc_id [str]: doc_id of the expected document to show up with the query
        retriever: SentenceSearcher class
        n_returns [int]: number of negative samples to retrieve for each query
        label [int]: label to assign paragraphs (1=correct, 0=neutral, -1=confirmed nonmatch)
    Returns:
        [List[Dict[str,str]]]: list of dictionaries of negative sample paragraphs
    """

    checked_results = []
    try:
        results = retriever.retrieve_topn(query, n_returns)
        logger.info(f"Retrieved {str(len(results))} negative samples for query: {query} / doc: {doc_id}")
        for result in results:
            par = data[data["paragraph_id"]==result['id']].iloc[0]["text"]
            par = ' '.join(par.split(' ')[:400])
            if check_no_match(doc_id, result['id']):
                checked_results.append({"query": query, "doc": result['id'], "paragraph": par, "label": 0})
    except Exception as e:
        logger.warning("Could not get negative paragraphs")
        logger.warning(e)
    
    return checked_results

def add_gold_standard(intel: Dict[str,str], gold_standard_path: Union[str, os.PathLike]) -> Dict[str,str]:
    """Adds original gold standard data to the intel training data.
    Args:
        intel [Dict[str,str]: intelligent search evaluation data
        gold_standard_path [Union[str, os.PathLike]]: path to load in the manually curated gold_standard.csv
    Returns:
        intel [Dict[str,str]: intelligent search evaluation data with manual entries added
    """
    gold = pd.read_csv(gold_standard_path, names=['query', 'document'])
    gold['query_clean'] = gold['query'].apply(lambda x: normalize_query(x))
    gold['docs_split'] = gold['document'].apply(lambda x: x.split(';'))
    all_docs = list(set([a for b in gold['docs_split'].tolist() for a in b]))

    def add_key(mydict: Dict[str,str]) -> str:
        """Adds new key to queries/collections dictionaries"""
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

def train_test_split(data: Dict[str,str], tts_ratio: float) -> Tuple[Dict[str, str]]:
    """Splits a dictionary into train/test set based on split ratio"""

    train_size = round(len(data) * tts_ratio)
    train_keys = random.sample(data.keys(), train_size)
    test_keys = [i for i in data.keys() if i not in train_keys]

    train = {k: data[k] for k in train_keys}
    test = {k: data[k] for k in test_keys}

    return train, test

def collect_matches(
    data: pd.DataFrame, 
    nlp,
    relations: Dict[str, str],
    queries: Dict[str, str],
    collection: Dict[str, str],
    label: int,
    ) -> Tuple[Dict[str, str]]:
    """Gets matching paragraphs for each query/docid pair
    Args:
        data [pd.DataFrame]: data df with processed text at paragraph_id level for sent_index
        sim: SimilarityRanker class
        relations [Dict[str, str]]: dictionary of query:doc matches from intelligent search data
        queries [Dict[str, str]]: dictionary of query ids : query text from intelligent search data
        collection [Dict[str, str]]: dictionary of match ids : match text (doc ids) from intelligent search data
        label [int]: label to assign paragraphs (1=correct, 0=neutral, -1=confirmed nonmatch)
        n_matching [int]: number of matching paragraphs to retrieve for the expected doc
    Returns:
        [Tuple[Dict[str, str]]]: one dictionary of found search pairs, one dictionary of notfound search pairs
    """
    found = {}
    not_found = {}
    count = 0
    for i in relations.keys():
        count += 1
        logger.info(count)
        query = queries[i]
        for k in relations[i]:
            doc = collection[k]
            uid = str(i) + '_' + str(k) # backup UID, overwritten if there are results
            try:
                matching = get_best_paragraphs(data, query, doc, nlp)
                for match in matching:
                    uid =  str(i) + '_' + str(match['id'])
                    text = ' '.join(match['text'].split(' ')[:400]) # truncate to 400 tokens
                    found[uid] = {"query": query, "doc": doc, "paragraph": text, "label": label}
            except Exception as e:
                logger.warning("Could not get positive matches")
                logger.warning(e)
                not_found[uid] = {"query": query, "doc": doc, "label": label}
    return found, not_found

def collect_negative_samples(
    data: pd.DataFrame, 
    retriever, 
    n_returns: int,
    relations: Dict[str, str],
    queries: Dict[str, str],
    collection: Dict[str, str],
    ) -> Tuple[Dict[str, str]]:
    """Gets negative samples each query/docid pair
    Args:
        data [pd.DataFrame]: data df with processed text at paragraph_id level for sent_index
        retriever: SentenceSearcher class
        n_returns [int]: number of non-matching paragraphs to retrieve for each query
        relations [Dict[str, str]]: dictionary of query:doc matches from intelligent search data
        queries [Dict[str, str]]: dictionary of query ids : query text from intelligent search data
        collection [Dict[str, str]]: dictionary of match ids : match text (doc ids) from intelligent search data
        label [int]: label to assign paragraphs (1=correct, 0=neutral, -1=confirmed nonmatch)
    Returns:
        [Tuple[Dict[str, str]]]: one dictionary of found search pairs, one dictionary of notfound search pairs
    """
    found = {}
    not_found = {}
    for i in relations.keys():
        query = queries[i]
        for k in relations[i]:
            doc = collection[k]
            uid = str(i) + '_' + str(k) + '_neg' # backup UID, overwritten if there are results
            try:
                not_matching = get_negative_paragraphs(data=data, query=query, doc_id=k, retriever=retriever, n_returns=n_returns)
                for match in not_matching:
                    uid =  str(i) + '_' + str(match['doc'])
                    text = ' '.join(match['paragraph'].split(' ')[:400]) # truncate to 400 tokens
                    found[uid] = {"query": query, "doc": doc, "paragraph": text, "label": 0}
            except Exception as e:
                logger.warning(e)
                not_found[uid] = {"query": query, "doc": doc, "label": 0}
                
    return found, not_found

def make_training_data(
    index_path: Union[str, os.PathLike],
    n_returns: int,
    n_matching: int,
    level: str, 
    update_eval_data: bool, 
    retriever=None,
    sim_model_name: str=SIM_MODEL,
    transformers_dir: Union[str,os.PathLike]=LOCAL_TRANSFORMERS_DIR,
    gold_standard_path: Union[str,os.PathLike]=gold_standard_path,
    tts_ratio: float=tts_ratio,
    training_dir: Union[str,os.PathLike]=training_dir) -> Tuple[Dict[str,str]]:
    """Makes training data based on new user search history data
    Args:
        index_path [str|os.PathLike]: path to the sent index for retrieving the training data (should be most recent index)
        n_returns [int]: number of non-matching paragraphs to retrieve for each query
        n_matching [int]: number of matching paragraphs to retrieve for the expected doc
        level [str]: level of eval tier to use for training data (options: ['all', 'silver', 'gold'])
        update_eval_data [bool]: whether or not to update the eval data before making training data
        sim_model_name [str]: name of sim model for loading SimilarityRanker
        transformers_dir [Union[str,os.PathLike]]: directory of transformer models
        gold_standard_path [Union[str,os.PathLike]]: path to load in the manually curated gold_standard.csv
        tts_ratio [float]: train/test split ratio, float from 0-1
        training_dir [Union[str,os.PathLike]]: directory for saving training data
    Returns:
        [Tuple[Dict[str,str]]]: training data and training metadata dictionaries
    """    
    ## open json files
    if not os.path.exists(os.path.join(DATA_PATH, "validation", "domain", "sent_transformer")) or update_eval_data:
        logger.info("****    Updating the evaluation data")
        make_tiered_eval_data(index_path)
    validation_dir = get_most_recent_dir(os.path.join(DATA_PATH, "validation", "domain", "sent_transformer"))
    directory = os.path.join(validation_dir, level)
    logger.info(f"****    Loading in intelligent search data from {str(directory)}")
    try:
        f = open_json('intelligent_search_data.json', directory)
        intel = json.loads(f)
    except Exception as e:
        logger.warning("Could not load intelligent search data")
        logger.warning(e)

    ## add gold standard samples
    logger.info("****   Adding gold standard examples")
    intel = add_gold_standard(intel, gold_standard_path)

    ## set up save dir
    save_dir = make_timestamp_directory(training_dir)

    try:
        nlp = spacy.load("en_core_web_lg")
    except:
        logger.warning("Could not load spacy model")

    if not retriever:
        logger.info("Did not init SentenceSearcher, loading now")
        retriever = SentenceSearcher(
            sim_model_name=sim_model_name, 
            index_path=index_path, 
            transformer_path=transformers_dir
            )
    ## read in sent_index data
    logger.info("****   Loading in sent index data from retriever")
    try:
        data = retriever.data
        data['doc_id'] = data['paragraph_id'].apply(lambda x: x.split('.pdf')[0])
    except Exception as e:
        logger.info("Could not load in data from retriever")
        logger.warning(e)

    ## get matching paragraphs
    try:
        correct_found, correct_notfound = collect_matches(
        data=data, queries=intel['queries'], collection=intel['collection'],
        relations=intel['correct'], label=1, nlp = nlp)
        logger.info(f"---Number of correct query/result pairs that were not found: {str(len(correct_notfound))}")
    except Exception as e:
        logger.warning(e)
        logger.warning("\nCould not retrieve positive matches\n")
    try:
        incorrect_found, incorrect_notfound = collect_matches(
        data=data, queries=intel['queries'], collection=intel['collection'],
        relations=intel['incorrect'], label=-1, nlp = nlp)
        logger.info(f"---Number of incorrect query/result pairs that were not found: {str(len(incorrect_notfound))}")
    except Exception as e:
        logger.warning(e)
        logger.warning("\nCould not retrieve negative matches\n")

    ## get negative samples
    try:
        all_relations = {**intel['correct'], **intel['incorrect']}
        neutral_found, neutral_notfound = collect_negative_samples(
        data=data, retriever=retriever, n_returns=n_returns, queries=intel['queries'], collection=intel['collection'],
        relations=all_relations)
        logger.info(f"---Number of negative sample pairs that were not found: {str(len(neutral_notfound))}")
    except Exception as e:
        logger.warning(e)
        logger.warning("\nCould not retrieve negative samples\n")

    ## save a json of the query-doc pairs that did not retrieve an ES paragraph for training data
    notfound = {**correct_notfound, **incorrect_notfound, **neutral_notfound}
    logger.info(f"---Number of total query/result pairs that were not found: {str(len(notfound))}")
    notfound_path = os.path.join(save_dir, 'not_found_search_pairs.json')
    with open(notfound_path, "w") as outfile:
        json.dump(notfound, outfile)

    ## train/test split (separate on correct/incorrect for balance)
    correct_train, correct_test = train_test_split(correct_found, tts_ratio)
    incorrect_train, incorrect_test = train_test_split(incorrect_found, tts_ratio)
    neutral_train, neutral_test = train_test_split(neutral_found, tts_ratio)
    train = {**correct_train, **incorrect_train, **neutral_train}
    test = {**correct_test, **incorrect_test, **neutral_test}

    try:## check labels
        pos = len([i for i in train if train[i]['label'] == 1])
        logger.info(f"*** {str(pos)} positive samples in TRAIN")
        neutral = len([i for i in train if train[i]['label'] == 0])
        logger.info(f"*** {str(neutral)} neutral samples in TRAIN")
        neg = len([i for i in train if train[i]['label'] == -1])
        logger.info(f"*** {str(neg)} negative samples in TRAIN")

        ## check labels
        pos_test = len([i for i in test if test[i]['label'] == 1])
        logger.info(f"*** {str(pos_test)} positive samples in TEST")
        neutral_test = len([i for i in test if test[i]['label'] == 0])
        logger.info(f"*** {str(neutral_test)} neutral samples in TEST")
        neg_test = len([i for i in test if test[i]['label'] == -1])
        logger.info(f"*** {str(neg_test)} negative samples in TEST")
    except Exception as e:
        logger.warning("Could not check stats for train/test")
        logger.warning(e)

    data = {"train": train, "test": test}
    metadata = {
        "date_created": str(date.today()),
        "n_positive_samples": f"{str(pos)} train / {str(pos_test)} test",
        "n_neutral_samples": f"{str(neutral)} train / {str(neutral_test)} test",
        "n_negative_samples": f"{str(neg)} train / {str(neg_test)} test",
        "train_size": len(train),
        "test_size": len(test),
        "split_ratio": tts_ratio
    }

    logger.info(f"**** Generated training data: \n {metadata}")

    ## save data and metadata files
    data_path = os.path.join(save_dir, 'training_data.json')
    metadata_path = os.path.join(save_dir, 'training_metadata.json')

    with open(data_path, "w") as outfile:
        json.dump(data, outfile)

    with open(metadata_path, "w") as outfile:
        json.dump(metadata, outfile)

if __name__ == '__main__':

    make_training_data(
        index_path="gamechangerml/models/sent_index_20210715", n_returns=50, n_matching=3, level="silver", 
        update_eval_data=False)
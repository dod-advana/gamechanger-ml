import argparse
import random
from gamechangerml.configs.config import TrainingConfig, ValidationConfig, EmbedderConfig, SimilarityConfig
from gamechangerml.src.search.sent_transformer.model import SentenceEncoder, SentenceSearcher
from gamechangerml.src.utilities.es_search_utils import connect_es, collect_results
from gamechangerml.src.utilities.test_utils import *
from gamechangerml.src.utilities.model_helper import *
from gamechangerml.api.utils.logger import logger
from gamechangerml.api.utils.pathselect import get_model_paths

model_path_dict = get_model_paths()
random.seed(42)

ES_URL = 'https://vpc-gamechanger-iquxkyq2dobz4antllp35g2vby.us-east-1.es.amazonaws.com'
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
BASE_MODEL_NAME = EmbedderConfig.MODEL_ARGS['model_name']
VALIDATION_DIR = get_most_recent_dir(os.path.join(ValidationConfig.DATA_ARGS['validation_dir'], 'sent_transformer'))
SENT_INDEX = model_path_dict["sentence"]
base_dir=TrainingConfig.DATA_ARGS['training_data_dir']
tts_ratio=TrainingConfig.DATA_ARGS['train_test_split_ratio']

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
    transformer_path=LOCAL_TRANSFORMERS_DIR, 
    encoder_config=EmbedderConfig.MODEL_ARGS, 
    similarity_config=SimilarityConfig.MODEL_ARGS, 
    use_gpu=False
    ):
    
    ## get query to collections dict
    newdict = {}
    reverse_q = {}
    reverse_r = {}
    for i in intel['correct'].keys():
        query = intel['queries'][i]
        reverse_q[query] = i
        res = intel['correct'][i]
        answers = []
        for j in res:
            ans = intel['collection'][j]
            reverse_r[ans] = j
            answers.append(ans)
        newdict[query] = answers
    
    ## look up queries against index
    neg_samples = {}
    encoder = SentenceEncoder(encoder_config, index_path, use_gpu)
    retriever = SentenceSearcher(index_path, transformer_path, encoder_config, similarity_config)
    for key in newdict:
        doc_texts, doc_ids, doc_scores = retriever.retrieve_topn(key)
        neg_samples[key] = [d for d in doc_ids if d not in newdict[key]]

    ## reverse lookup queries to collections
    final_dict = {}
    for i in newdict.keys():
        query = reverse_q[i]
        answers = []
        for j in newdict[i]:
            res = reverse_r[j]
            answers.append(res)
        final_dict[query] = answers
    
    intel['incorrect'].update(final_dict)

    return

def make_training_data(base_dir, tts_ratio):

    ## open json files
    directory = os.path.join(VALIDATION_DIR, 'any')
    f = open_json('intelligent_search_data.json', directory)
    intel = json.loads(f)

    ## collect negative samples
    lookup_negative_samples(intel)
    
    ## set up save dir
    sub_dir = os.path.join(base_dir, 'sent_transformer')
    save_dir = make_timestamp_directory(sub_dir)

    ## query ES
    es = connect_es(ES_URL)
    correct_found, correct_notfound = collect_results(relations=intel['correct'], queries=intel['queries'], collection=intel['collection'], es=es, label=1)
    incorrect_found, incorrect_notfound = collect_results(relations=intel['incorrect'], queries=intel['queries'], collection=intel['collection'], es=es, label=0)

    ## save a df of the query-doc pairs that did not retrieve an ES paragraph for training data
    notfound = {**correct_notfound, **incorrect_notfound}
    notfound_path = os.path.join(save_dir, timestamp_filename('not_found_search_pairs', '.json'))
    with open(notfound_path, "w") as outfile:
        json.dump(notfound, outfile)

    ## train/test split (separate on correct/incorrect for balance)
    correct_train, correct_test = train_test_split(correct_found, tts_ratio)
    incorrect_train, incorrect_test = train_test_split(incorrect_found, tts_ratio)
    train = {**correct_train, **incorrect_train}
    test = {**correct_test, **incorrect_test}

    data = {"train": train, "test": test}
    metadata = {
        "date_created": str(date.today()),
        "n_positive_samples": len(correct_found),
        "n_negative_samples": len(incorrect_found),
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

    make_training_data(base_dir=base_dir, tts_ratio=tts_ratio)
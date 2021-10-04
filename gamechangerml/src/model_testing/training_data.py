import pandas as pd
import json
import os
from datetime import date

from gamechangerml.src.model_testing.validation_data import IntelSearchData
from gamechangerml.src.utilities.es_search_utils import connect_es, collect_results
from gamechangerml.src.utilities.model_helper import timestamp_filename, open_json, make_timestamp_directory, CustomJSONizer
from gamechangerml.configs.config import ValidationConfig, EmbedderConfig, TrainingConfig
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.api.utils.logger import logger
import random

model_path_dict = get_model_paths()

ES_URL = 'https://vpc-gamechanger-iquxkyq2dobz4antllp35g2vby.us-east-1.es.amazonaws.com'
VALIDATION_DIR = ValidationConfig.DATA_ARGS['validation_dir']
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
BASE_MODEL_NAME = EmbedderConfig.MODEL_ARGS['model_name']

random.seed(42)

class TrainingData():

    def __init__(self, base_dir, train_test_split_ratio):

        self.base_dir = base_dir
        self.tts_ratio = train_test_split_ratio

    def train_test_split(self, data):
        '''Splits a dictionary into train/test set based on split ratio'''

        train_size = round(len(data) * self.tts_ratio)
        train_keys = random.sample(data.keys(), train_size)
        test_keys = [i for i in data.keys() if i not in train_keys]

        train = {k: data[k] for k in train_keys}
        test = {k: data[k] for k in test_keys}

        return train, test

class SentenceTransformerTD(TrainingData):

    def __init__(
        self, 
        trigger_es,
        max_results,
        min_correct_matches,
        start_date,
        end_date,
        exclude_searches,
        base_dir, 
        train_test_split_ratio, 
        ):

        super().__init__(base_dir, train_test_split_ratio)
        #self.sub_dir = os.path.join(self.base_dir, 'sent_transformer')
        self.sub_dir = os.path.join('gamechangerml/data/training', 'sent_transformer')
        self.start_date = start_date
        self.end_date = end_date
        self.min_correct_matches = min_correct_matches
        self.max_results = max_results
        self.trigger_es = trigger_es
        self.exclude_searches=exclude_searches
        self.gold_data, self.gold_metadata = self.make_training_data(level='gold')
        self.silver_data, self.silver_metadata = self.make_training_data(level='silver')

    def make_training_data(self, level):
        '''Make training data JSON for first time, if there isn't one'''

        ## make gold intel
        save_dir = make_timestamp_directory(self.sub_dir)
        if level == 'gold': #use passed in args for gold level data
            min_correct_matches=self.min_correct_matches, 
            max_results=self.max_results
        elif level == 'silver':
            min_correct_matches=TrainingConfig.DATA_ARGS['silver_level']['min_correct_matches']
            max_results=TrainingConfig.DATA_ARGS['silver_level']['min_correct_matches']

        intel = IntelSearchData(
            validation_dir=ValidationConfig.DATA_ARGS['validation_dir'],
            matamo_dir=ValidationConfig.DATA_ARGS['matamo_dir'],
            search_hist_dir=ValidationConfig.DATA_ARGS['search_hist_dir'],
            start_date=self.start_date, 
            end_date=self.end_date,
            exclude_searches=self.exclude_searches,
            min_correct_matches=min_correct_matches, 
            max_results=max_results, 
            )

        save_intel = {
            "queries": intel.queries, 
            "collection": intel.collection, 
            "meta_relations": intel.all_relations,
            "correct": intel.correct,
            "incorrect": intel.incorrect}

        save_intel = json.dumps(save_intel, cls=CustomJSONizer)
        intel_path = os.path.join(save_dir, level, 'intelligent_search_data.json')
        with open(intel_path, "w") as outfile:
            json.dump(save_intel, outfile)
        
        ## TODO: Meka eval data

        ## get data from ES
        if self.trigger_es:
            es = connect_es(ES_URL)
            correct_found, correct_notfound = collect_results(relations=intel.correct, queries=intel.queries, collection=intel.collection, es=es, label=1)
            incorrect_found, incorrect_notfound = collect_results(relations=intel.incorrect, queries=intel.queries, collection=intel.collection, es=es, label=0)

            ## save a df of the query-doc pairs that did not retrieve an ES paragraph for training data
            notfound = {**correct_notfound, **incorrect_notfound}
            notfound_path = os.path.join(save_dir, timestamp_filename('not_found_search_pairs', '.json'))
            with open(notfound_path, "w") as outfile:
                json.dump(notfound, outfile)

            ## train/test split (separate on correct/incorrect for balance)
            correct_train, correct_test = self.train_test_split(correct_found)
            incorrect_train, incorrect_test = self.train_test_split(incorrect_found)
            train = {**correct_train, **incorrect_train}
            test = {**correct_test, **incorrect_test}

            data = {"train": train, "test": test}
            metadata = {
                "date_created": str(date.today()),
                "n_positive_samples": len(correct_found),
                "n_negative_samples": len(incorrect_found),
                "train_size": len(train),
                "test_size": len(test),
                "split_ratio": self.tts_ratio
            }

            ## save data and metadata files
            data_path = os.path.join(save_dir, 'training_data.json')
            metadata_path = os.path.join(save_dir, 'training_metadata.json')

            with open(data_path, "w") as outfile:
                json.dump(data, outfile)

            with open(metadata_path, "w") as outfile:
                json.dump(metadata, outfile)

            return data, metadata

        else:
            return
import spacy
from gamechangerml.src.text_handling.process import preprocess
import numpy as np
import re
from gamechangerml.src.search.ranking import search_data as meta
from gamechangerml.src.search.ranking import rank
from gamechangerml import REPO_PATH
import datetime
import pandas as pd
from tqdm import tqdm
import argparse
import logging
import os
from elasticsearch import Elasticsearch
import pickle
import xgboost as xgb
import graphviz
import matplotlib
import math
import requests
from sklearn.preprocessing import LabelEncoder


ES_HOST = "https://vpc-gamechanger-dev-es-ms4wkfqyvlyt3gmiyak2hleqyu.us-east-1.es.amazonaws.com"

client = Elasticsearch([ES_HOST])
logger = logging.getLogger("gamechanger")


class LTR:
    def __init__(
        self,
        params={
            "max_depth": 10,
            "eta": 0.3,
            "silent": 0,
            "objective": "rank:map",
            "num_round": 10,
        },
    ):
        self.data = self.read_xg_data()
        self.params = params
        self.mappings = self.read_mappings()

    def write_model(self, model):
        with open("xgb-model.json", "w") as output:
            output.write("[" + ",".join(list(model)) + "]")
            output.close()

    def read_xg_data(self, path="xgboost.txt"):
        try:
            self.data = xgb.DMatrix(path)
            return self.data
        except Exception as e:
            logger.error("Could not read in data for training")

    def read_mappings(self, path="gamechangerml/data/SearchPdfMapping.csv"):
        try:
            self.mappings = pd.read_csv(path)
        except Exception as e:
            logger.error("Could not read in mappings to make judgement list")
        return self.mappings

    def train(self, write=True):
        bst = xgb.train(self.params, self.data)
        model = bst.get_dump(fmap="featmap.txt", dump_format="json")
        if write:
            self.write_model(model)
        return bst, model

    def post_model(self, model, model_name):
        query = {
            "model": {
                "name": model_name,
                "model": {"type": "model/xgboost+json", "definition": model},
            }
        }
        endpoint = ES_HOST + "/_ltr/_featureset/doc_features/_createmodel?pretty"
        r = requests.post(endpoint, data=query)
        return r

    def generate_judgement(self, mappings):
        searches = mappings[["search", "document"]]
        searches.dropna(inplace=True)
        searches.search.replace("&quot;", "", regex=True, inplace=True)
        word_tuples = []
        for row in tqdm(searches.itertuples()):
            words = row.search.split(" ")
            for word in words:
                clean = word.lower()
                clean = re.sub(r"[^\w\s]", "", clean)
                clean = preprocess(clean, remove_stopwords=True)
                if clean:
                    tup = (clean[0], row.document)
                    word_tuples.append(tup)
        tuple_df = pd.DataFrame(word_tuples, columns=["search", "document"])
        count_df = pd.DataFrame()
        for keyword in tuple_df.search.unique():
            a = tuple_df[tuple_df.search == keyword]
            tmp_df = a.groupby("document").count()
            tmp_df["keyword"] = keyword
            count_df = count_df.append(tmp_df)
        count_df.sort_values("search")
        arr = count_df.search.copy()
        count_df["ranking"] = self.normalize(arr)
        count_df.ranking = count_df.ranking.apply(np.ceil)
        count_df.ranking = count_df.ranking.astype(int)
        le = LabelEncoder()
        count_df["qid"] = le.fit_transform(count_df.keyword)

        return count_df

    def query_es_fts(self, df):
        ltr_log = []
        print("querying es ltr logs")
        for kw in tqdm(df.keyword.unique()):
            tmp = df[df.keyword == kw]
            for docs in tmp.itertuples():
                doc = docs.Index
                q = self.construct_query(doc, kw)
                r = client.search(index="gamechanger", body=dict(q))
                ltr_log.append(r["hits"]["hits"])
        return ltr_log

    def process_ltr_log(self, ltr_log, num_fts=4):
        all_vals = []
        print("processing logs")
        for entries in ltr_log:
            if len(entries) > 0:
                print(entries)
                # loop through entry logs (num of features)
                fts = []
                for entry in entries[0]["fields"]["_ltrlog"][0]["log_entry1"]:
                    if "value" in entry:
                        fts.append(entry["value"])
                    else:
                        fts.append(0)
                all_vals.append(fts)
            else:
                all_vals.append(np.zeros(num_fts))
        return all_vals

    def generate_ft_txt_file(self, df):
        ltr_log = self.query_es_fts(df)
        vals = self.process_ltr_log(ltr_log)
        ft_df = pd.DataFrame(
            vals, columns=["title", "kw", "textlength", "paragraph"])
        df.reset_index(inplace=True)
        df = pd.concat([df, ft_df], axis=1)

        print("generating txt file")
        for kw in tqdm(df.keyword.unique()):
            rows = df[df.keyword == kw]
            for i in rows.itertuples():
                new_row = (
                    str(int(i.ranking))
                    + " qid:"
                    + str(i.qid)
                    + " 1:"
                    + str(i.title)
                    + " 2:"
                    + str(i.paragraph)
                    + " 3:"
                    + str(i.kw)
                    + " 4:"
                    + str(i.textlength)
                    + " # "
                    + kw
                    + " "
                    + str(i.document)
                    + "\n"
                )
                with open("xgboost.txt", "a") as f:
                    f.writelines(new_row)
        return df

    def construct_query(self, doc, kw):
        query = {
            "_source": ["filename", "fields"],
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"filename": [doc]}},
                        {
                            "sltr": {
                                "_name": "logged_featureset",
                                "featureset": "doc_features",
                                "params": {"keywords": kw},
                            }
                        },
                    ]
                }
            },
            "ext": {
                "ltr_log": {
                    "log_specs": {
                        "name": "log_entry1",
                        "named_query": "logged_featureset",
                    }
                }
            },
        }
        return query

    def post_features(self):
        query = {
            "featureset": {
                "name": "doc_features",
                "features": [
                    {
                        "name": "1",
                        "params": ["keywords"],
                        "template_language": "mustache",
                        "template": {
                            "wildcard": {
                                "display_title_s.search": {
                                    "value": "*{{keywords}}*",
                                    "boost": 2,
                                }
                            }
                        },
                    },
                    {
                        "name": "2",
                        "params": ["keywords"],
                        "template_language": "mustache",
                        "template": {
                            "nested": {
                                "path": "paragraphs",
                                "inner_hits": {},
                                "query": {
                                    "bool": {
                                        "should": [
                                            {
                                                "query_string": {
                                                    "query": "{{keywords}}",
                                                    "default_field": "paragraphs.par_raw_text_t.gc_english",
                                                    "default_operator": "AND",
                                                    "fuzzy_max_expansions": 1000,
                                                    "fuzziness": "AUTO",
                                                    "analyzer": "gc_english",
                                                }
                                            }
                                        ]
                                    }
                                },
                            }
                        },
                    },
                ],
            }
        }

    def normalize(self, arr, start=1, end=4):
        width = end - start
        res = (arr - arr.min()) / (arr.max() - arr.min()) * width + start
        return res

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


ES_HOST = "https://vpc-gamechanger-dev-es-ms4wkfqyvlyt3gmiyak2hleqyu.us-east-1.es.amazonaws.com"

client = Elasticsearch([ES_HOST])
logger = logging.getLogger("gamechanger")


class LTR:
    def __init__(
        self,
        data="xgboost.txt",
        params={
            "max_depth": 6,
            "eta": 0.3,
            "silent": 0,
            "objective": "rank:pairwise",
            "num_round": 10,
        },
        num_round=5,
    ):
        self.data = xgb.DMatrix(data)
        self.params = params
        self.num_round = num_round

    def write_model(self, model):
        with open("xgb-model.json", "w") as output:
            output.write("[" + ",".join(list(model)) + "]")
            output.close()

    def train(self, write=True):
        bst = xgb.train(self.params, self.data, self.num_round)
        model = bst.get_dump(fmap="featmap.txt", dump_format="json")
        if write:
            self.write_model(model)
        return bst, model

    def generate_judgement(self):
        df = pd.read_csv("gamechangerml/data/SearchPdfMapping.csv")
        searches = df[["search", "document"]]
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
        count_df["norm"] = self.normalize(arr)
        count_df.norm = count_df.norm.apply(np.ceil)
        count_df.sort_values("norm")
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

    def process_ltr_log(self, ltr_log):
        text_vals = []
        title_vals = []
        print("processing logs")
        for x in ltr_log:
            if len(x) > 0:
                text_log = x[0]["fields"]["_ltrlog"][0]["log_entry1"][1]
                title_log = x[0]["fields"]["_ltrlog"][0]["log_entry1"][0]
                if "value" in text_log:
                    text_vals.append(text_log["value"])
                else:
                    text_vals.append(0)
                if "value" in title_log:
                    title_vals.append(title_log["value"])
                else:
                    title_vals.append(0)
            else:
                text_vals.append(0)
                title_vals.append(0)
        return title_vals, text_vals

    def generate_ft_txt_file(self, df):
        ltr_log = query_es_fts(df)
        title_vals, text_vals = self.process_ltr_log(ltr_log)
        df["title_vals"] = title_vals
        df["text_vals"] = text_vals

        print("generating txt file")
        count = 0
        for kw in tqdm(df.keyword.unique()):
            rows = df[df.keyword == kw]
            for i in rows.itertuples():
                new_row = (
                    str(int(i.norm))
                    + " qid:"
                    + str(count)
                    + " 1:"
                    + str(i.title_vals)
                    + " 2:"
                    + str(i.text_vals)
                    + " # "
                    + kw
                    + " "
                    + i.Index
                    + "\n"
                )
                count += 1
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

    def normalize(self, arr, start=1, end=4):
        width = end - start
        res = (arr - arr.min()) / (arr.max() - arr.min()) * width + start
        return res

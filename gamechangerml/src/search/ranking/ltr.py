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
import matplotlib
import math
import requests
from sklearn.preprocessing import LabelEncoder


ES_HOST = "https://vpc-gamechanger-dev-es-ms4wkfqyvlyt3gmiyak2hleqyu.us-east-1.es.amazonaws.com"

client = Elasticsearch([ES_HOST])
logger = logging.getLogger("gamechanger")
GC_MODEL_PATH = "gamechangerml/models/ltr"
GC_DATA_PATH = "gamechangerml/data/ltr"


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
        self.judgement = None

    def write_model(self, model):
        """write model: writes model to file
        params: model in json form
        returns:
        """
        # write model to json for LTR
        path = os.path.join(GC_MODEL_PATH, "xgb-model.json")
        if not os.path.exists(GC_MODEL_PATH):
            os.mkdir(GC_MODEL_PATH)
        with open(path, "w") as output:
            output.write("[" + ",".join(list(model)) + "]")
            output.close()

    def read_xg_data(self, path=os.path.join(GC_DATA_PATH, "xgboost.txt")):
        """read xg data: reads LTR formatted data
        params: path to file
        returns:
        """
        try:
            self.data = xgb.DMatrix(path)
            return self.data
        except Exception as e:
            logger.error("Could not read in data for training")

    def read_mappings(self, path="gamechangerml/data/SearchPdfMapping.csv"):
        """read mappings: reads search pdf mappings
        params: path to file
        returns:
            mappings file
        """
        try:
            self.mappings = pd.read_csv(path)
        except Exception as e:
            logger.error("Could not read in mappings to make judgement list")
        return self.mappings

    def train(self, write=True):
        """train - train a xgboost model with parameters
        params:
            write: boolean to write to file
        returns:
            bst: xgboost object
            model: model json
        """
        bst = xgb.train(self.params, self.data)
        model = bst.get_dump(
            fmap=os.path.join(GC_DATA_PATH, "featmap.txt"), dump_format="json"
        )
        if write:
            self.write_model(model)
        return bst, model

    def post_model(self, model, model_name):
        """post model - post a model to ES
        params:
            model: model in json form
            model_name: model name for ES
        returns:
            r: results
        """
        query = {
            "model": {
                "name": model_name,
                "model": {"type": "model/xgboost+json", "definition": model},
            }
        }
        endpoint = ES_HOST + "/_ltr/_featureset/doc_features/_createmodel"
        r = requests.post(endpoint, data=query)
        return r

    def search(self, terms, rescore=True):
        """search: searches with a rescore with ltr option
        params:
            terms: search terms
            rescore: boolean
        returns:
            r: results
        """
        query = {
            "_source": {"includes": ["pagerank_r", "kw_doc_score_r"]},
            "stored_fields": ["filename", "title"],
            "from": 0,
            "size": 15,
            "query": {
                "bool": {
                    "must": [],
                    "should": [
                        {
                            "nested": {
                                "path": "paragraphs",
                                "inner_hits": {},
                                "query": {
                                    "bool": {
                                        "should": [
                                            {
                                                "query_string": {
                                                    "query": f"{terms}",
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
                        {
                            "multi_match": {
                                "query": f"{terms}",
                                "fields": ["display_title_s.search"],
                                "type": "phrase",
                                "operator": "and",
                                "boost": 4,
                            }
                        },
                        {"wildcard": {"keyw_5": {"value": f"*{terms}*"}}},
                        {
                            "wildcard": {
                                "display_title_s.search": {
                                    "value": f"*{terms}*",
                                    "boost": 6,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                    "filter": [{"term": {"is_revoked_b": "false"}}],
                }
            },
            "highlight": {
                "fields": {"display_title_s.search": {}, "keyw_5": {}, "id": {}},
                "fragmenter": "simple",
            },
            "sort": [{"_score": {"order": "desc"}}],
        }
        if rescore:
            query["rescore"] = {
                "query": {
                    "rescore_query": {
                        "sltr": {
                            "params": {"keywords": f"{terms}"},
                            "model": "lambda_rank1",
                        }
                    }
                }
            }
        r = client.search(index="gamechanger", body=dict(query))
        return r

    def generate_judgement(self, mappings):
        """generate judgement - generates judgement list from user mapping data
        params:
            mappings: dataframe of user data extracted from pdf mapping table
        returns:
            count_df: cleaned dataframe with search mapped data
        """
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
        self.judgement = count_df

        return count_df

    def query_es_fts(self, df):
        """query ES features: gets ES feature logs from judgement list
        params:
            df: dataframe of judgement list and keyword
        returns:
            ltr_log: logs of from ES
        """
        ltr_log = []
        logger.info("querying es ltr logs")
        # loop through all unique keywords
        for kw in tqdm(df.keyword.unique()):
            # get frame of all of the keyword rows
            tmp = df[df.keyword == kw]
            # get logged feature
            for docs in tmp.itertuples():
                doc = docs.Index
                q = self.construct_query(doc, kw)
                r = client.search(index="gamechanger", body=dict(q))
                ltr_log.append(r["hits"]["hits"])
        return ltr_log

    def process_ltr_log(self, ltr_log, num_fts=4):
        """process ltr log: extracts features from ES logs for judgement list
        params:
            ltr_log: results from ES
            num_fts: number of features
        returns:
            all_vals: all logged features in matrix
        """
        all_vals = []
        logger.info("processing logs")
        for entries in ltr_log:
            if len(entries) > 0:
                # loop through entry logs (num of features)
                fts = []
                for entry in entries[0]["fields"]["_ltrlog"][0]["log_entry1"]:
                    # checks if entry is empty
                    if "value" in entry:
                        fts.append(entry["value"])
                    else:
                        fts.append(0)
                all_vals.append(fts)
            # if the doc doesnt exist then add all 0s
            else:
                all_vals.append(np.zeros(num_fts))
        return all_vals

    def generate_ft_txt_file(self, df):
        """generate feature text file: creates the LTR formatted training data
        params:
            df: dataframe of the judgement list with features
        returns:
            outputs a file
        """
        ltr_log = self.query_es_fts(df)
        vals = self.process_ltr_log(ltr_log)
        ft_df = pd.DataFrame(
            vals, columns=["title", "kw", "textlength", "paragraph"])
        df.reset_index(inplace=True)
        df = pd.concat([df, ft_df], axis=1)

        logger.info("generating txt file")
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
        """construct query: constructs query for logging features from es
        params:
            doc: document name that is in corpus
            kw: keyword to search on
        returns: query
        """
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
        """post features: post features to es"""
        query = {
            "featureset": {
                "name": "doc_features",
                "features": [
                    {
                        "name": "title",
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
                        "name": "keyw_5",
                        "params": ["keywords"],
                        "template_language": "mustache",
                        "template": {"match": {"keyw_5": "{{keywords}}"}},
                    },
                    {
                        "name": "textlength",
                        "params": ["keywords"],
                        "template_language": "mustache",
                        "template": {
                            "function_score": {
                                "functions": [
                                    {
                                        "field_value_factor": {
                                            "field": "page_count",
                                            "missing": 0,
                                        }
                                    }
                                ],
                                "query": {"match_all": {}},
                            }
                        },
                    },
                    {
                        "name": "paragraph",
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

    def normalize(self, arr, start=0, end=4):
        """normalize: basic normalize between two numbers function
        params:
            arr: array to normalize
            start: beginning number integer
            end: ending number integer
        returns: normalized array
        """
        width = end - start
        res = (arr - arr.min()) / (arr.max() - arr.min()) * width + start
        return res

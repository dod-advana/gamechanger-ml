# from gamechangerml.src.search.ranking import matamo as mt
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

ES_HOST = "https://vpc-gamechanger-dev-es-ms4wkfqyvlyt3gmiyak2hleqyu.us-east-1.es.amazonaws.com"

client = Elasticsearch([ES_HOST])
"""
Usage:
    example:
    python -m gamechangerml.src.search.ranking.generate_ft -c test/small_corpus/ -dd 80 --prod gamechangerml/src/search/ranking/generated_files/prod_test_data.csv

optional arguements:
    --corpus, -c Corpus directory
    --days -dd days since today to get data
"""


logger = logging.getLogger("gamechanger")

corpus_dir = "test/corpus_new"
prod_data_file = os.path.join(
    REPO_PATH, "gamechangerml/src/search/ranking/generated_files/prod_test_data.csv"
)


def generate_pop_docs(pop_kw_df: pd.DataFrame, corpus_df: pd.DataFrame) -> pd.DataFrame:
    """generate popular documents based on keyword searches
    Args:
        pop_kw_df: dataframe of keywords and counts
        corpus_df: dataframe of corpus unique ID with text
    Returns:
        dataframe

    """

    docList = []
    for row_kw in tqdm(pop_kw_df.itertuples()):
        for row_corp in corpus_df.itertuples():
            if len(row_corp.keywords):
                if row_kw.keywords in row_corp.keywords[0]:
                    docList.append(
                        {"id": row_corp.id, "keywords": row_kw.keywords})
    docDf = pd.DataFrame(docList, columns=["id", "keywords"])
    docCounts = docDf.groupby("id").count().sort_values(
        "keywords", ascending=False)
    docCounts.rename(columns={"keywords": "kw_in_doc_score"}, inplace=True)

    return docCounts


def generate_ft_doc(corpus_dir: str, days: int = 80, prod_data: str = prod_data_file):
    """generate feature document
    Args:
        corpus_dir: corpus directory
        days: how many days to retrieve data
    Returns:

    """
    today = datetime.datetime.now()
    out_dir = os.path.join(
        REPO_PATH, "gamechangerml/src/search/ranking/generated_files"
    )
    r = rank.Rank()
    day_delta = 80
    d = datetime.timedelta(days=day_delta)
    from_date = today - d

    # TELEMETRY
    # tele_df = mt.get_telemetry(day_delta)
    # kw_doc_pairs = mt.parse_onDocOpen(tele_df)
    # kw_doc_df = pd.DataFrame(kw_doc_pairs)

    # SEARCH LOGS
    # resp = meta.get_searchLogs(str(from_date.date()))

    # until we get connection to prod data
    resp = pd.read_csv(prod_data)
    popular_keywords = meta.get_top_keywords(resp)
    meta_df = meta.scored_logs(resp)

    # CORPUS
    corp_df = r._getCorpusData(corpus_dir)
    pr_df = r.get_pr_docs(corpus_dir)
    corp_df = corp_df.merge(pr_df)

    docCounts = generate_pop_docs(popular_keywords, corp_df)
    corp_df = corp_df.merge(docCounts, on="id", how="outer")

    corp_df["kw_in_doc_score"] = corp_df["kw_in_doc_score"].fillna(0.00001)
    corp_df["kw_in_doc_score"] = (
        corp_df["kw_in_doc_score"] - corp_df["kw_in_doc_score"].min()
    ) / (corp_df["kw_in_doc_score"].max() - corp_df["kw_in_doc_score"].min())
    corp_df.kw_in_doc_score.loc[corp_df.kw_in_doc_score == 0] = 0.00001
    corp_df.to_csv(os.path.join(out_dir, "corpus_meta.csv"))


def generate_judgement():
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
    count_df["norm"] = normalize(arr)
    count_df.norm = count_df.norm.apply(np.ceil)
    count_df.sort_values("norm")
    return count_df


def query_es_fts(df):
    ltr_log = []
    print("querying es ltr logs")
    for kw in tqdm(df.keyword.unique()):
        tmp = df[df.keyword == kw]
        for docs in tmp.itertuples():
            doc = docs.Index
            q = construct_query(doc, kw)
            r = client.search(index="gamechanger", body=dict(q))
            ltr_log.append(r["hits"]["hits"])
    return ltr_log


def process_ltr_log(ltr_log):
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


def generate_ft_txt_file(df):
    ltr_log = query_es_fts(df)
    title_vals, text_vals = process_ltr_log(ltr_log)
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


def construct_query(doc, kw):
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
                "log_specs": {"name": "log_entry1", "named_query": "logged_featureset"}
            }
        },
    }
    return query


def normalize(arr, start=1, end=4):
    width = end - start
    res = (arr - arr.min()) / (arr.max() - arr.min()) * width + start
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Features CSV")
    parser.add_argument(
        "--corpus", "-c", dest="corpus_dir", help="corpus directory, full path"
    )
    parser.add_argument(
        "--days",
        "-dd",
        dest="day_delta",
        default=80,
        help="days of data to grab since todays date",
    )
    # Until we can pull data from postgres from production automatically (currently avail in dev)
    parser.add_argument(
        "--prod",
        "-p",
        dest="prod_data",
        default=os.path.join(
            REPO_PATH,
            "gamechangerml/src/search/ranking/generated_files/prod_test_data.csv",
        ),
        help="production data historical search logs csv ",
    )

    # parser.add_argument(
    #    "--outfile", "-o", dest="outfile", help="generated outfile name"
    # )

    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    days = args.day_delta
    prod_data = args.prod_data
    # outfilename = args.outfile
    # generate_ft_doc(corpus_dir=corpus_dir, days=days, prod_data=prod_data)

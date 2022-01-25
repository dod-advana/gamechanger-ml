from elasticsearch import Elasticsearch
import json
import requests
import re
import pandas as pd
import os
import logging
import time
from gamechangerml import MODEL_PATH, DATA_PATH
import typing as t
import base64
from urllib.parse import urljoin

logger = logging.getLogger("gamechanger")


class ESUtils:
    def __init__(
        self,
        host: str = os.environ.get("ES_HOST", "localhost"),
        port: str = os.environ.get("ES_PORT", 443),
        user: str = os.environ.get("ES_USER", ""),
        password: str = os.environ.get("ES_PASSWORD", ""),
        enable_ssl: bool = os.environ.get(
            "ES_ENABLE_SSL", "True").lower() == "true",
        enable_auth: bool = os.environ.get(
            "ES_ENABLE_AUTH", "False").lower() == "true",
        es_index: str = os.environ.get("ES_INDEX", "gamechanger"),
    ):

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.enable_ssl = enable_ssl
        self.enable_auth = enable_auth
        self.es_index = es_index

        self.auth_token = base64.b64encode(
            f"{self.user}:{self.password}".encode()
        ).decode()

    @property
    def client(self) -> Elasticsearch:
        if hasattr(self, "_client"):
            return getattr(self, "_client")

        host_args = dict(
            hosts=[
                {
                    "host": self.host,
                    "port": self.port,
                    "http_compress": True,
                    "timeout": 60,
                }
            ]
        )
        auth_args = (
            dict(http_auth=(self.user, self.password)
                 ) if self.enable_auth else {}
        )
        ssl_args = dict(use_ssl=self.enable_ssl)

        es_args = dict(
            **host_args,
            **auth_args,
            **ssl_args,
        )

        self._es_client = Elasticsearch(**es_args)
        return self._es_client

    @property
    def auth_headers(self) -> t.Dict[str, str]:
        return {"Authorization": f"Basic {self.auth_token}"} if self.enable_auth else {}

    @property
    def content_headers(self) -> t.Dict[str, str]:
        return {"Content-Type": "application/json"}

    @property
    def default_headers(self) -> t.Dict[str, str]:
        if self.enable_auth:
            return dict(**self.auth_headers, **self.content_headers)
        else:
            return dict(**self.content_headers)

    @property
    def root_url(self) -> str:
        return ("https" if self.enable_ssl else "http") + f"://{self.host}:{self.port}/"

    def request(self, method: str, url: str, **request_opts) -> requests.Response:
        complete_url = urljoin(self.root_url, url.lstrip("/"))
        return requests.request(
            method=method,
            url=complete_url,
            headers=self.default_headers,
            **request_opts,
        )

    def post(self, url: str, **request_opts) -> requests.Response:
        return self.request(method="POST", url=url, **request_opts)

    def put(self, url: str, **request_opts) -> requests.Response:
        return self.request(method="PUT", url=url, **request_opts)

    def get(self, url: str, **request_opts) -> requests.Response:
        return self.request(method="GET", url=url, **request_opts)

    def delete(self, url: str, **request_opts) -> requests.Response:
        return self.request(method="DELETE", url=url, **request_opts)


def connect_es(es_url):
    """Connect to ES"""

    tries = 0
    while tries < 5:
        try:
            es = Elasticsearch([es_url])
            time.sleep(1)
            print("ES connected\n")
            break
        except ConnectionError:
            print("ES not connected, trying again\n")
            tries += 1

    return es


def get_es_responses_doc(es, query, docid):
    """Query ES for a search string and a docid (from search results)"""

    true = True
    false = False

    search = {
        "_source": {
            "includes": ["pagerank_r", "kw_doc_score_r", "orgs_rs", "topics_rs"]
        },
        "stored_fields": [
            "filename",
            "title",
            "page_count",
            "doc_type",
            "doc_num",
            "ref_list",
            "id",
            "summary_30",
            "keyw_5",
            "p_text",
            "type",
            "p_page",
            "display_title_s",
            "display_org_s",
            "display_doc_type_s",
            "is_revoked_b",
            "access_timestamp_dt",
            "publication_date_dt",
            "crawler_used_s",
        ],
        "from": 0,
        "size": 20,
        "track_total_hits": true,
        "query": {
            "bool": {
                "must": [
                    {"match": {"id": docid}},
                    {
                        "nested": {
                            "path": "paragraphs",
                            "inner_hits": {
                                "_source": false,
                                "stored_fields": [
                                    "paragraphs.page_num_i",
                                    "paragraphs.filename",
                                    "paragraphs.par_raw_text_t",
                                ],
                                "from": 0,
                                "size": 5,
                                "highlight": {
                                    "fields": {
                                        "paragraphs.filename.search": {
                                            "number_of_fragments": 0
                                        },
                                        "paragraphs.par_raw_text_t": {
                                            "fragment_size": 200,
                                            "number_of_fragments": 1,
                                        },
                                    },
                                    "fragmenter": "span",
                                },
                            },
                            "query": {
                                "bool": {
                                    "should": [
                                        {
                                            "wildcard": {
                                                "paragraphs.filename.search": {
                                                    "value": query,
                                                    "boost": 15,
                                                }
                                            }
                                        },
                                        {
                                            "query_string": {
                                                "query": query,
                                                "default_field": "paragraphs.par_raw_text_t",
                                                "default_operator": "AND",
                                                "fuzzy_max_expansions": 100,
                                                "fuzziness": "AUTO",
                                            }
                                        },
                                    ]
                                }
                            },
                        }
                    },
                ],
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "keyw_5^2",
                                "id^2",
                                "summary_30",
                                "paragraphs.par_raw_text_t",
                            ],
                            "operator": "or",
                        }
                    },
                    {"rank_feature": {"field": "pagerank_r", "boost": 0.5}},
                    {"rank_feature": {"field": "kw_doc_score_r", "boost": 0.1}},
                ],
            }
        },
    }

    return es.search(index="gamechanger", body=search)


def get_paragraph_results(es, query, doc):
    """Get list of paragraph texts for each search result"""

    docid = doc + ".pdf_0"
    resp = get_es_responses_doc(es, query, docid)

    texts = []
    if resp["hits"]["total"]["value"] > 0:
        hits = resp["hits"]["hits"][0]["inner_hits"]["paragraphs"]["hits"]["hits"]
        for par in hits:
            texts.append(par["fields"]["paragraphs.par_raw_text_t"])

    return texts


def collect_results(relations, queries, collection, es, label):
    """Query ES for search/doc matches and add them to query results with a label"""

    found = {}
    not_found = {}
    for i in relations.keys():
        query = queries[i]
        for k in relations[i]:
            doc = collection[k]
            uid = str(i) + "_" + str(k)
            try:
                para = get_paragraph_results(es, query, doc)[0][0]
                # truncate to 400 tokens
                para = " ".join(para.split(" ")[:400])
                found[uid] = {
                    "query": query,
                    "doc": doc,
                    "paragraph": para,
                    "label": label,
                }
            except:
                not_found[uid] = {"query": query, "doc": doc, "label": label}

    return found, not_found

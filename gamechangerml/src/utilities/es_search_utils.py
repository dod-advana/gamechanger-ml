from elasticsearch import Elasticsearch
import json, requests
import re
import pandas as pd
import time

def connect_es(es_url):
    '''Connect to ES'''

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
    '''Query ES for a search string and a docid (from search results)'''
    
    true = True
    false = False
    
    search = {
          "_source": {
            "includes": [
              "pagerank_r",
              "kw_doc_score_r",
              "orgs_rs",
              "topics_rs"
            ]
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
            "crawler_used_s"
          ],
          "from": 0,
          "size": 20,
          "track_total_hits": true,
          "query": {
            "bool": {
              "must": [
                {
                  "match": {
                    "id": docid
                  }
                },
                {
                  "nested": {
                    "path": "paragraphs",
                    "inner_hits": {
                      "_source": false,
                      "stored_fields": [
                        "paragraphs.page_num_i",
                        "paragraphs.filename",
                        "paragraphs.par_raw_text_t"
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
                            "number_of_fragments": 1
                          }
                        },
                        "fragmenter": "span"
                      }
                    },
                    "query": {
                      "bool": {
                        "should": [
                          {
                            "wildcard": {
                              "paragraphs.filename.search": {
                                "value": query,
                                "boost": 15
                              }
                            }
                          },
                          {
                            "query_string": {
                              "query": query,
                              "default_field": "paragraphs.par_raw_text_t",
                              "default_operator": "AND",
                              "fuzzy_max_expansions": 100,
                              "fuzziness": "AUTO"
                            }
                          }
                        ]
                      }
                    }
                  }
                }
              ],
              "should": [
                {
                  "multi_match": {
                    "query": query,
                    "fields": [
                      "keyw_5^2",
                      "id^2",
                      "summary_30",
                      "paragraphs.par_raw_text_t"
                    ],
                    "operator": "or"
                  }
                },
                {
                  "rank_feature": {
                    "field": "pagerank_r",
                    "boost": 0.5
                  }
                },
                {
                  "rank_feature": {
                    "field": "kw_doc_score_r",
                    "boost": 0.1
                  }
                }
              ]
            }
          }
        }
    
    return es.search(index="gamechanger", body=search)

def get_paragraph_results(es, query, doc):
    '''Get list of paragraph texts for each search result'''
    
    docid = doc + '.pdf_0'
    resp = get_es_responses_doc(es, query, docid)
    
    texts = []
    if resp['hits']['total']['value'] > 0:
        hits = resp['hits']['hits'][0]['inner_hits']['paragraphs']['hits']['hits']
        for par in hits:
            texts.append(par['fields']['paragraphs.par_raw_text_t'])
    
    return texts

def collect_results(relations, queries, collection, es, label):
    '''Query ES for search/doc matches and add them to query results with a label'''

    found = {}
    not_found = {}
    for i in relations.keys():
        query = queries[i]
        for k in relations[i]:
            doc = collection[k]
            uid = str(i) + '_' + str(k)
            try:
                para = get_paragraph_results(es, query, doc)[0][0]
                para = ' '.join(para.split(' ')[:400]) # truncate to 400 tokens
                found[uid] = {"query": query, "doc": doc, "paragraph": para, "label": label}
            except:
                not_found[uid] = {"query": query, "doc": doc, "label": label}
                
    return found, not_found
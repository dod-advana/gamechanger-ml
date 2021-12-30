import numpy as np
import pandas as pd 
import os
import csv
from collections import Counter
import networkx as nx
from gamechangerml.api.utils.logger import logger
from gamechangerml.src.utilities.test_utils import open_json
from gamechangerml import DATA_PATH

CORPUS_DIR = "/Users/katherinedowdy/Desktop/parsed_docs"

def get_file_data(filename, df):

    try:
        filename = filename.strip('.pdf')
        idx = df.index[df['Name']==filename].tolist()[0]
        row = df.loc[idx]
        return row.to_dict()
    except Exception as e:
        logger.warning(f"Couldn't retrieve data for file: {filename}")
        logger.warning(e)
        return None

class Recommender:

    def __init__(self, corpus_dir=CORPUS_DIR, update=[], reload_data=False):

        self.graph = self.get_user_graph()
        self.data = self.load_data(corpus_dir, update, reload_data)

    def load_data(self, corpus_dir, update, reload_data):

        data_path = os.path.join(DATA_PATH, "recommender", "data.csv") 
        if os.path.exists(data_path) and reload_data == True:
            logger.info(" ****    BUILDING RECOMMENDER: Reading in recommender data")
            return pd.read_csv(data_path)
        else:
            logger.info(" ****    BUILDING RECOMMENDER: Making data file")
            return self.update_data(corpus_dir, update)

    def _get_corpus_meta(self, corpus_dir, update):

        try:
            corpus_file = os.path.join(DATA_PATH, "recommender", "corpus_data.csv")
            if os.path.isfile(corpus_file) and "corpus_data" not in update:
                logger.info(f" ****    BUILDING RECOMMENDER: Reading in corpus data from {str(corpus_file)}")
                df = pd.read_csv(corpus_file)
            else:
                logger.info(" ****    BUILDING RECOMMENDER: Making corpus metadata csv")
                jsons = [i for i in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, i))]
                columns = ['filename', 'title', 'display_org_s', 'display_doc_type_s', 'publication_date_dt', 'ref_list']
                with open(corpus_file, 'w') as csvfile:
                    csvwriter = csv.writer(csvfile)  
                    csvwriter.writerow(columns)
                    for i in range(len(jsons)):
                        file = jsons[i]
                        logger.info(f"Collecting document {str(i)} / {str(len(jsons))}: {file}")
                        doc = open_json(file, corpus_dir)
                        row = [[
                            doc['filename'],
                            doc['title'],
                            doc['display_org_s'],
                            doc['display_doc_type_s'],
                            doc['publication_date_dt'],
                            doc['ref_list']
                        ]]
                        csvwriter.writerows(row)
                df = pd.read_csv(corpus_file)
            df['Name'] = df['filename'].apply(lambda x: x.strip('.pdf'))
            df.fillna('', inplace = True)
            return df[['Name', 'title', 'display_org_s', 'display_doc_type_s', 'publication_date_dt']]
        
        except Exception as e:
            logger.warning("Could not get corpus metadata")
            logger.warning(e)
            return pd.DataFrame()

    def _get_label_prop(self, update):

        logger.info(" ****    BUILDING RECOMMENDER: Retrieving label propagation clusters") 
        try:
            label_prop_file = os.path.join(DATA_PATH, "recommender", "label_prop.csv")
            if os.path.exists(label_prop_file) and "clusters" not in update:
                lp = pd.read_csv(label_prop_file)
            else:
                ## QUERY NEO4J
                lp = None

            lp['total'] = lp['Community'].map(lp.groupby('Community').count().to_dict()['Name'])
            lp.rename(columns = {'Community': 'label_prop', 'total': 'label_prop_total'}, inplace = True)
            return lp
        except Exception as e:
            logger.warning("Could not retrieve label propagation clusters")
            logger.warning(e)
            return pd.DataFrame()
    
    def _get_louvain(self, update):
        
        logger.info(" ****    BUILDING RECOMMENDER: Retrieving louvain clusters")
        try:
            louvain_file = os.path.join(DATA_PATH, "recommender", "louvain.csv")
            if os.path.exists(louvain_file) and "clusters" not in update:
                louv = pd.read_csv(louvain_file)
            else:
                ## QUERY NEO4J
                louv = None
            louv.rename(columns = {'communityId': 'louv_final', 'name': 'Name'}, inplace = True)
            louv['louv_total'] = louv['louv_final'].map(louv.groupby('louv_final').count().to_dict()['Name'])
            return louv
        except Exception as e:
            logger.warning("Could not retrieve louvain cluster assignments")
            logger.warning(e)
            return pd.DataFrame()

    def _get_betweenness(self, update):

        logger.info(" ****    BUILDING RECOMMENDER: Getting betweenness scores")
        try:
            btw_file = os.path.join(DATA_PATH, "recommender", "betweenness.csv")
            if os.path.exists(btw_file) and "betweenness" not in update:
                btw = pd.read_csv(btw_file)
            else:
                ## QUERY NEO
                btw = None
            btw.rename(columns = {'name': 'Name'}, inplace = True)
            return btw
        except Exception as e:
            logger.warning("Could not retrieve betweenness scores")
            logger.warning(e)
            return pd.DataFrame()
    
    def get_user_graph(self):

        logger.info(" ****    BUILDING RECOMMENDER: Making user graph")
        try:
            user_file = os.path.join(DATA_PATH, "user_data", "search_history", "SearchPdfMapping.csv")
            user = pd.read_csv(user_file)
            user.dropna(subset = ['document'], inplace = True)
            user['clean_search'] = user['search'].apply(lambda x: x.replace('&quot;', '"'))
            user['clean_doc'] = user['document'].apply(lambda x: x.replace(",,", ","))
            pairs = [(x, y) for y, x in zip(user['clean_doc'], user['clean_search'])]
            user_graph = nx.Graph()
            user_graph.add_edges_from(pairs)

            return user_graph
        except Exception as e:
            logger.warning("Could not make user graph")
            logger.warning(e)
            return nx.Graph()
    
    def update_data(self, corpus_dir, update):
        
        logger.info("****    Updating recommender data")
        try:
            lp = self._get_label_prop(update)
            louv = self._get_louvain(update)
            btw = self._get_betweenness(update)
            corpus_data = self._get_corpus_meta(corpus_dir, update)

            merged = lp.merge(louv, on=['Name'])
            merged = btw.merge(merged, on=['Name'])
            merged = corpus_data.merge(merged, on=['Name'])

            merged['betweenness_perc'] = merged['followers'].rank(pct = True)
            merged['org_doc'] = merged['display_org_s'] + ': ' + merged['display_doc_type_s']
            merged['fullname'] = merged['Name'] + ' (' + merged['title'] + ')'
            path = os.path.join(DATA_PATH, "recommender", "data.csv")
            merged.to_csv(path)
            return merged
        except Exception as e:
            logger.warning("Could not update recommender data")
            logger.warning(e)
            return pd.DataFrame()

    def _lookup_cluster(self, doc, rank_method=['importance', 'nearness']):
    
        try:
            if doc['label_prop_total'] < doc['louv_total']:
                col = 'label_prop'
                community = doc['label_prop']
            else:
                col = 'louv_final'
                community = doc['louv_final']
            sort = self.data[self.data[col]==community].copy()
            sort = sort.sort_values(by='followers', ascending=False).reset_index()
            idx = sort.index[sort['Name']==doc['Name']].tolist()[0]
            if rank_method == 'importance':
                result = sort.head(6)   
            elif rank_method == 'nearness':
                r = idx - 3
                z = idx + 3
                result = sort.loc[r:z]
            else:
                logger.warn("Did not pass a valid rank method: options are 'importance' and 'nearness'")
                result = sort.head(6)   
            try:
                res = result.drop(idx)
            except:
                res = result
            return [res.loc[i].to_dict() for i in res.index]
        except Exception as e:
            logger.warning("Could not look up group clusters for this file")
            logger.warning(e)
            return []

    def _lookup_history(self, filename):

        try:
            if filename.strip('.pdf') == filename:
                filename = filename + '.pdf'
            searches = list(self.graph.adj[filename])
            related = []
            for i in searches:
                rels = list(self.graph.adj[i])
                rels = [x for x in rels if x != filename]
                if rels != []:
                    related.extend(rels)
            logger.info(f"Found {len(set(related))} documents opened with same searches")
            related.sort()
            counts = Counter(related)
            top = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]}
            return [get_file_data(f.strip('.pdf'), self.data) for f in list(top.keys())]
        
        except Exception as e:
            logger.warning("Could not lookup docs from similar searches")
            logger.warning(e)
            return []
    
    def _lookup_other(self, filename):

        try:
            filename = filename.strip('.pdf')
            dd = self.data.sort_values(by = ['org_doc', 'display_org_s', 'publication_date_dt']).reset_index().copy()
            idx = dd.index[dd['Name']==filename].tolist()[0]
            start = int(idx) - 3
            end = int(idx) + 3
            sub = dd.loc[start:end]
            if idx in sub.index.tolist():
                sub = sub.drop(idx)
            return [sub.loc[i].to_dict() for i in sub.index]
        except Exception as e:
            logger.warning("Could not collect backup cluster results")
            logger.warning(e)
            return []
    
    def get_recs(self, filename, rank_method='importance', verbose=True):

        if filename in list(self.data['Name']):
            doc = get_file_data(filename, self.data)
            biggest_group = np.max([doc['louv_total'], doc['label_prop_total']])
            betweenness = np.round(float(doc['betweenness_perc'])*100, 3)
            if verbose:
                logger.info(f"*** Looking up {filename}: \n {str(doc)} ***\n")
                if betweenness > self.data['betweenness_perc'].min()*100 + 1:
                    logger.info(f"This doc is more important than {betweenness}% of docs \
                        (based on betweenness)")
                else:
                    logger.info(f"This doc is one of the least connected docs in the \
                        corpus based on betweenness (same as ~{betweenness}% of docs)\n")
                
            results = self._lookup_history(filename)
            method = 'search_history'
            if len(results) < 1:
                if biggest_group > 1:
                    logger.info("Didn't find the the document in user search history; getting clusters.") 
                    results = self._lookup_cluster(doc, rank_method)
                    method = 'clusters'
                else:
                    results = self._lookup_other(filename)
                    method = 'backup'
            logger.info(f"method: {method}, \nresults: {str(results)}")
            return {"method": method, "results": results}
        else:
            logger.warning("This document is not in the corpus")
            return {}
import os
import numpy as np
import pandas as pd
import csv
from gamechangerml.src.search.sent_transformer.model import SentenceEncoder, SentenceSearcher, SimilarityRanker
from gamechangerml.src.search.QA.QAReader import DocumentReader as QAReader
from gamechangerml.configs.config import EmbedderConfig, SimilarityConfig, ValidationConfig
from gamechangerml.src.utilities.model_helper import *
from gamechangerml.src.model_testing.validation_data import SQuADData, NLIData, MSMarcoData, QADomainData, RetrieverDomainData
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.api.utils.logger import logger


model_path_dict = get_model_paths()
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
SENT_INDEX_PATH = model_path_dict["sentence"]
SAVE_PATH = ValidationConfig.DATA_ARGS['evaluation_dir']

class TransformerEvaluator():

    def __init__(self, transformer_path=LOCAL_TRANSFORMERS_DIR, save_path=SAVE_PATH):

        self.transformer_path = transformer_path
        self.save_path = check_directory(save_path)
        ## check if model was last changed?
        ## check ig validation data has changed?
        ## if model or validation data is new, run eval again
        ## read in + add to old stats?

class QAEvaluator(TransformerEvaluator):

    def __init__(
        self, 
        model_name, 
        qa_type, 
        nbest, 
        null_threshold, 
        transformer_path=LOCAL_TRANSFORMERS_DIR, 
        save_path=SAVE_PATH, 
        new_model=False, 
        new_data=False,
        use_gpu=False,
        sample_limit=ValidationConfig.DATA_ARGS['squad']['sample_limit']):

        super().__init__(transformer_path, save_path)

        self.model_path = os.path.join(transformer_path, model_name)
        self.model = QAReader(self.model_path, qa_type, nbest, null_threshold, use_gpu)
        self.squad = SQuADData()
        self.domain = QADomainData()
        self.sample_limit = sample_limit
        if new_model == True:
            self.squad_results = self.eval(test_data='squad')
            self.domain_results = self.eval(test_data='domain')
        elif new_data == True:
            self.domain_results = self.eval(test_data='domain')

    def compare(self, predicted, actual):

        clean_pred = normalize_answer(predicted)
        clean_answers = set([normalize_answer(i['text']) for i in actual])
        if clean_pred in clean_answers:
            return True
        else:
            return False

    def predict(self, test_data=['squad', 'domain']):

        columns = [
            'index',
            'queries',
            'actual_answers',
            'predicted_answer',
            'actual_null',
            'predicted_null',
            'answers_match',
            'nulls_match'
        ]

        query_count = 0
        if test_data == 'squad':
            queries = self.squad.queries
        elif test_data == 'domain':
            queries = self.domain.checked_queries

        csv_filename = os.path.join(self.save_path, timestamp_filename(test_data, '.csv'))
        with open(csv_filename, 'w') as csvfile: 
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(columns) 

            for query in queries:
                try:
                    print(query_count, query['question'])
                    actual_null = query['null_expected']
                    actual = query['expected']
                    prediction = self.model.answer(query['question'], query['search_context'])[0]
                    if prediction['status'] == 'failed' or prediction['text'] == '':
                        predicted_null = True
                    else:
                        predicted_null = False
                    answer_match = self.compare(prediction['text'], query['expected'])
                    null_match = bool(actual_null == predicted_null)
                
                    row = [[
                            str(query_count),
                            str(query['question']),
                            str(actual),
                            str(prediction),
                            str(actual_null),
                            str(predicted_null),
                            str(answer_match),
                            str(null_match),
                        ]]
                    csvwriter.writerows(row)
                    query_count += 1
                except:
                    break

        return pd.read_csv(csv_filename)

    def eval(self, test_data):

        df = self.predict(test_data)

        ## change to squad metrics with tokens
        #df['true_neg'] = np.where(df['actual_null']==True and df['predicted_null'] == True, True, False)
        #df['true_pos'] = np.where(df['actual_null']==False and df['answers_match'] == True, True, False)
        #df['false_neg'] = np.where(df['predicted_null']==True and df['answers_match'] == False, True, False)
        #df['false_pos'] = np.where(df['predicted_null']==False and df['answers_match'] == False, True, False)

        num_queries = df['queries'].nunique()
        proportion_answers_match = np.round(df['answers_match'].value_counts(normalize = True)[True], 2)
        proportion_nulls_match = np.round(df['nulls_match'].value_counts(normalize = True)[True], 2)

        agg_results = {
            "num_queries": num_queries,
            "proportion_answer_match": proportion_answers_match,
            "proportion_null_match": proportion_nulls_match,
        }
        return agg_results

class RetrieverEvaluator(TransformerEvaluator):

    def __init__(
            self, 
            model_name=EmbedderConfig.MODEL_ARGS['model_name'],
            transformer_path=LOCAL_TRANSFORMERS_DIR,
            index_path=None,
            save_path=SAVE_PATH, 
            encoder_args=EmbedderConfig.MODEL_ARGS, 
            similarity_args=SimilarityConfig.MODEL_ARGS,
            corpus_path=None,
            use_gpu=False
        ):

        super().__init__(transformer_path, save_path)

        self.model_path = os.path.join(transformer_path, model_name)
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.data = None
        if self.index_path:
            if not os.path.exists(self.index_path):  
                os.makedirs(self.index_path)
                self.encoder = SentenceEncoder(encoder_args, self.index_path, use_gpu)
                self.make_index()
            self.retriever = SentenceSearcher(self.index_path, transformer_path, encoder_args, similarity_args)
        
    def make_index(self):

        return self.encoder.index_documents(corpus_path = self.corpus_path)

    def predict(self):

        columns = [
            'index',
            'queries',
            'top_expected_ids',
            'predicted_rank',
            'text',
            'top_result_match',
            'in_top_10',
            'score'
        ]

        csv_filename = os.path.join(self.save_path, timestamp_filename('msmarco_eval', '.csv'))
        with open(csv_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(columns) 

            query_count = 0
            for idx, query in self.data.queries.items(): 
                rank = 'NA'
                matching_text = 'NA'
                score = 'NA'
                top_result_match = False
                in_top_10 = False
                print(query_count, query)
                expected_id = self.data.relations[idx][0]
                doc_texts, doc_ids, doc_scores = self.retriever.retrieve_topn(query)
                if expected_id in doc_ids:
                    in_top_10 = True
                    rank = doc_ids.index(expected_id)
                    matching_text = self.data.collection[expected_id]
                    score = doc_scores[rank]
                    if rank == 0:
                        top_result_match = True

                row = [[
                    str(query_count),
                    str(query),
                    str(expected_id),
                    str(rank),
                    str(matching_text),
                    str(top_result_match),
                    str(in_top_10),
                    str(score)
                ]]
                csvwriter.writerows(row)
                query_count += 1

        return pd.read_csv(csv_filename)
        
    def eval(self):
        
        df = self.predict()

        num_queries = df['queries'].nunique()
        proportion_expected_in_top_10 = np.round(df['in_top_10'].value_counts(normalize = True)[True], 2)
        proportion_expected_is_top = np.round(df['top_result_match'].value_counts(normalize = True)[True], 2)
        agg_results = {
            "num_queries": num_queries,
            "proportion_expected_in_top_10": proportion_expected_in_top_10,
            "proportion_expected_is_top": proportion_expected_is_top
        }
        return agg_results

class MSMarcoEvaluator(RetrieverEvaluator):

    def __init__(
            self, 
            model_name=EmbedderConfig.MODEL_ARGS['model_name'],
            transformer_path=LOCAL_TRANSFORMERS_DIR,
            index_path='msmarco_index',
            save_path=SAVE_PATH, 
            encoder_args=EmbedderConfig.MODEL_ARGS, 
            similarity_args=SimilarityConfig.MODEL_ARGS,
            corpus_path=None,
            use_gpu=False,
            new_model=False
        ):

        super().__init__(model_name, transformer_path, save_path, encoder_args, similarity_args, corpus_path, use_gpu)
    
        self.data = MSMarcoData()
        self.index_path = os.path.join(save_path, index_path)
        if new_model == True:
            self.results = self.eval()

class GoldStandardRetrieverEvaluator(RetrieverEvaluator):

    def __init__(
            self, 
            model_name=EmbedderConfig.MODEL_ARGS['model_name'],
            transformer_path=LOCAL_TRANSFORMERS_DIR,
            index_path=SENT_INDEX_PATH,
            save_path=SAVE_PATH, 
            encoder_args=EmbedderConfig.MODEL_ARGS, 
            similarity_args=SimilarityConfig.MODEL_ARGS,
            corpus_path=None,
            use_gpu=False,
            new_model=False,
            new_data=True
        ):

        super().__init__(model_name, transformer_path, save_path, encoder_args, similarity_args, corpus_path, use_gpu)
    
        self.data = RetrieverDomainData()
        self.index_path = index_path
        if new_model == True:
            self.results = self.eval()
        elif new_data == True:
            self.results = self.eval()


class SimilarityEvaluator(TransformerEvaluator):

    def __init__(
            self, 
            transformer_path=LOCAL_TRANSFORMERS_DIR, 
            save_path=SAVE_PATH, 
            model_args=SimilarityConfig.MODEL_ARGS, 
            new_model = False,
            sample_limit=1000
        ):

        super().__init__(transformer_path, save_path)

        self.model = SimilarityRanker(model_args, transformer_path)
        self.nli = NLIData()
        self.sample_limit = sample_limit
        if new_model == True:
            self.agg_results = self.eval_nli()

    def predict_nli(self):

        df = self.nli.sample_csv
        ranks = {}
        count = 0
        cutoff = np.min([df['promptID'].nunique(), self.sample_limit])
        for i in df['promptID'].unique():
            if count <= cutoff:
                print(count, i)
                subset = df[df['promptID']==i]
                iddict = dict(zip(subset['sentence2'], subset['pairID']))
                texts = [i for i in iddict.keys()]
                ids = [i for i in iddict.values()]
                query = self.nli.query_lookup[i]
                rank = 0
                for result in self.model.re_rank(query, texts, ids):
                    match_id = result['id']
                    match = result['text']
                    ranks[match_id] = rank
                    rank +=1

                count += 1
            else:
                break
        
        df['predicted_rank'] = df['pairID'].map(ranks)
        df.dropna(subset = ['predicted_rank'], inplace = True)
        df['predicted_rank'] = df['predicted_rank'].map(int)
        df['match'] = np.where(df['predicted_rank']==df['expected_rank'], True, False)

        return df

    def eval_nli(self):

        # create csv of predictions
        df = self.predict_nli()
        csv_filename = os.path.join(self.save_path, timestamp_filename('nli_eval', '.csv'))
        df.to_csv(csv_filename)

        # get overall stats
        proportion_all_match = np.round(df['match'].value_counts(normalize = True)[True], 2)
        proportion_top_match = np.round(df[df['expected_rank']==0]['match'].value_counts(normalize = True)[True], 2)
        num_queries = df['promptID'].nunique()

        agg_results = {
            "num_queries": num_queries,
            "proportion_all_match": proportion_all_match,
            "proportion_top_match": proportion_top_match
        }
        return agg_results
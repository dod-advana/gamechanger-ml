import os
import numpy as np
import pandas as pd
from gamechangerml.src.search.sent_transformer.model import SentenceEncoder, SentenceSearcher, SimilarityRanker
from gamechangerml.src.search.QA.QAReader import DocumentReader as QAReader
from gamechangerml.configs.config import QAConfig, EmbedderConfig, SimilarityConfig
from gamechangerml.src.utilities.model_helper import *
from gamechangerml.src.model_testing.validation_data import SQuADData, NLIData, MSMarcoData
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.api.utils.logger import logger


model_path_dict = get_model_paths()
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
SENT_INDEX_PATH = model_path_dict["sentence"]
SAVE_PATH = 'gamechangerml/data/evaluation'

class TransformerEvaluator():

    def __init__(self, transformer_path=LOCAL_TRANSFORMERS_DIR, save_path=SAVE_PATH):

        self.transformer_path = transformer_path
        self.save_path = save_path
        #self.date = datetime.today()

        ## check if model was last changed?
        ## check ig validation data has changed?
        ## if model or validation data is new, run eval again
        ## read in + add to old stats?

class QAEvaluator(TransformerEvaluator):

    def __init__(self, model_name, transformer_path, validation_path, qa_type, nbest, null_threshold, new_model=False, use_gpu=False):

        super().__init__(transformer_path, validation_path)

        self.model_path = os.path.join(transformer_path, model_name)
        self.model = QAReader(self.model_path, qa_type, nbest, null_threshold, use_gpu)
        if new_model == True:
            self.agg_results = self.eval_squad()

    def compare_squad(self, predicted, actual):

        clean_pred = normalize_answer(predicted)
        clean_answers = set([normalize_answer(i['text']) for i in actual])
        if clean_pred in clean_answers:
            return True
        else:
            return False

    def predict_squad(self):

        columns = ','.join([
            'queries',
            'actual_answers',
            'predicted_answer',
            'actual_null',
            'predicted_null',
            'answers_match',
            'nulls_match'
        ])

        query_count = 0
        csv_filename = os.path.join(self.save_path, timestamp_filename('squad_eval', '.csv'))
        eval_csv = open(csv_filename, "w")
        eval_csv.write(columns)

        queries = SQuADData.queries
        for query in queries:
            logger.info(query_count, query)
            query_count += 1
            actual_null = query['null_expected']
            actual = query['expected']
            prediction = self.model.answer(query['question'], query['search_context'])[0]
            if prediction['status'] == 'failed' or prediction['text'] == '':
                predicted_null = True
            else:
                predicted_null = False
            answer_match = self.compare_squad(prediction['text'], query['answers'])
            null_match = bool(actual_null == predicted_null)
        
            row = ','.join([
                    query,
                    actual,
                    prediction,
                    actual_null,
                    predicted_null,
                    answer_match,
                    null_match
                ])
            eval_csv.write(row)

        eval_csv.close()

        return pd.read(csv_filename)

    def eval_squad(self):

        df = self.predict_squad()

        ## change to squad metrics with tokens
        df['true_neg'] = np.where(df['actual_null']==True and df['predicted_null'] == True, True, False)
        df['true_pos'] = np.where(df['actual_null']==False and df['answers_match'] == True, True, False)
        df['false_neg'] = np.where(df['predicted_null']==True and df['answers_match'] == False, True, False)
        df['false_pos'] = np.where(df['predicted_null']==False and df['answers_match'] == False, True, False)

        num_queries = df['queries'].nunique()
        proportion_answers_match = np.round(df['answer_match'].value_counts(normalize = True)[True], 2)
        proportion_nulls_match = np.round(df['null_match'].value_counts(normalize = True)[True], 2)

        agg_results = {
            "num_queries": num_queries,
            "proportion_answer_match": proportion_answers_match,
            "proportion_null_match": proportion_nulls_match,
        }
        return agg_results

class MSMarcoEvaluator(TransformerEvaluator):

    def __init__(
            self, 
            model_name, 
            transformer_path, 
            save_path, 
            encoder_args=EmbedderConfig.MODEL_ARGS, 
            similarity_args=SimilarityConfig.MODEL_ARGS,
            use_gpu=False,
            new_model=False
        ):

        super().__init__(transformer_path, save_path)

        self.model_path = os.path.join(transformer_path, model_name)
        self.index_path = os.path.join(save_path, 'test_index')
        self.encoder = SentenceEncoder(encoder_args, self.index_path, use_gpu)
        self.retriever = SentenceSearcher(encoder_args, similarity_args)
        self.msmarco = MSMarcoData
        if new_model == True:
            self.agg_results = self.eval_msmarco()

    def make_msmarco_index(self):

        return self.encoder.index_documents(corpus_path = None)

    def predict_msmarco(self):

        columns = ','.join([
            'queries',
            'predicted_rank',
            'predicted_text',
            'expected_top_match',
            'top_result_match',
            'in_top_10',
            'score'
        ])
        query_count = 0
        csv_filename = os.path.join(self.save_path, timestamp_filename('msmarco_eval', '.csv'))
        eval_csv = open(csv_filename, "w")
        eval_csv.write(columns)

        for idx, query in self.msmarco.queries.items():
            logger.info(query_count, query)
            query_count += 1
            expected_text = self.msmarco.relations[idx]
            doc_texts, doc_ids, doc_scores = self.encoder.retrieve_topn(query)
            for _id in expected_text:
                idx = doc_ids.index(_id)
                predicted_rank = idx
                predicted_text = self.msmarco.collection[_id]
                score = doc_scores[idx]
                if _id == doc_ids[0]:
                    top_result_match = True
                elif _id in doc_ids:
                    top_result_match = False
                    in_top_10 = True
                else:
                    in_top_10 = False
                row = ','.join([
                    query,
                    predicted_rank,
                    predicted_text,
                    expected_text,
                    top_result_match,
                    in_top_10,
                    score
                ])
                eval_csv.write(row)

        eval_csv.close()

        return pd.read(csv_filename)
        
    def eval_msmarco(self):
        
        df = self.predict_msmarco()

        num_queries = df['queries'].nunique()
        proportion_expected_in_top_10 = np.round(df['in_top_10'].value_counts(normalize = True)[True], 2)
        proportion_expected_is_top = np.round(df['top_result_match'].value_counts(normalize = True)[True], 2)

        agg_results = {
            "num_queries": num_queries,
            "proportion_expected_in_top_10": proportion_expected_in_top_10,
            "proportion_expected_is_top": proportion_expected_is_top
        }
        return agg_results


class SimilarityEvaluator(TransformerEvaluator):

    def __init__(
            self, 
            transformer_path, 
            save_path, 
            model_name=SimilarityConfig.MODEL_ARGS['model_name'], 
            new_model = False
        ):

        super().__init__(transformer_path, save_path)

        self.model_path = os.path.join(transformer_path, model_name)
        self.model = SimilarityRanker(self.model_path)
        if new_model == True:
            self.agg_results = self.eval_nli()

    def predict_nli(self):

        df = NLIData.sample_csv
        ranks = {}
        for i in df['promptID'].unique():
            subset = df[df['promptID']==i]
            iddict = dict(zip(subset['sentence2'], subset['pairID']))
            texts = [iddict.keys()]
            ids = [iddict.values()]
            query = [subset['sentence1']][0]
            rank = 0
            for idx, score in self.model.re_rank(query, texts, ids):
                match = texts[idx]
                matchID = iddict[match]
                ranks[matchID] = rank
                rank +=1
        df['predicted_rank'] = df['pairID'].map(ranks)
        df['match'] = np.where(df['predicted_rank']==df['expected_rank'], True, False)

        return df

    def eval_nli(self):

        # create csv of predictions
        df = self.predict_nli()
        df.to_csv(os.path.join(self.save_path, timestamp_filename('nli_eval', '.csv')))

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
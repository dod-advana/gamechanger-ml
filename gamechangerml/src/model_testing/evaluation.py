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

class QAEvaluator(TransformerEvaluator):

    def __init__(
        self, 
        model_name, 
        qa_type, 
        nbest, 
        null_threshold, 
        transformer_path=LOCAL_TRANSFORMERS_DIR, 
        save_path=SAVE_PATH,
        use_gpu=False,
        ):

        super().__init__(transformer_path, save_path)

        self.model_path = os.path.join(transformer_path, model_name)
        self.model = QAReader(self.model_path, qa_type, nbest, null_threshold, use_gpu)

    def compare(self, predicted, actual):
        '''Compare predicted to expected answers'''

        clean_pred = normalize_answer(predicted)
        clean_answers = set([normalize_answer(i['text']) for i in actual])
        if clean_pred in clean_answers:
            return True
        else:
            return False

    def predict(self, data, test_data=['squad', 'domain']):
        '''Get answer predictions'''

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

        csv_filename = os.path.join(self.save_path, timestamp_filename(test_data, '.csv'))
        with open(csv_filename, 'w') as csvfile: 
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(columns) 

            for query in data.queries:
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

    def eval(self, data, test_data):
        '''Get evaluation stats across predicted/expected answer comparisons'''

        df = self.predict(data, test_data)

        num_queries = df['queries'].nunique()
        proportion_answers_match = np.round(df['answers_match'].value_counts(normalize = True)[True], 2)
        #proportion_nulls_match = np.round(df['nulls_match'].value_counts(normalize = True)[True], 2)

        agg_results = {
            "num_queries": num_queries,
            "proportion_exact_match": proportion_answers_match,
            #"proportion_null_match": proportion_nulls_match,
        }
        return agg_results

class SQuADQAEvaluator(QAEvaluator):

    def __init__(
        self, 
        model_name, 
        qa_type, 
        nbest, 
        null_threshold, 
        transformer_path=LOCAL_TRANSFORMERS_DIR, 
        save_path=SAVE_PATH, 
        new_model=False,
        use_gpu=False,
        ):

        super().__init__(model_name, qa_type, nbest, null_threshold, transformer_path, save_path, use_gpu)

        self.data = SQuADData()
        if new_model == True:
            self.results = self.eval(data=self.data, test_data='squad')

class IndomainQAEvaluator(QAEvaluator):

    def __init__(
        self, 
        model_name, 
        qa_type, 
        nbest, 
        null_threshold, 
        transformer_path=LOCAL_TRANSFORMERS_DIR, 
        save_path=SAVE_PATH, 
        new_model=False,
        use_gpu=False,
        ):

        super().__init__(model_name, qa_type, nbest, null_threshold, transformer_path, save_path, use_gpu)

        self.data = QADomainData()
        if new_model == True:
            self.results = self.eval(data=self.data, test_data='domain')


class RetrieverEvaluator(TransformerEvaluator):

    def __init__(
            self, 
            model_name=EmbedderConfig.MODEL_ARGS['model_name'],
            transformer_path=LOCAL_TRANSFORMERS_DIR,
            save_path=SAVE_PATH, 
        ):

        super().__init__(transformer_path, save_path)

        self.model_path = os.path.join(self.transformer_path, model_name)

    def make_index(self, encoder, corpus_path):

        return encoder.index_documents(corpus_path)

    def predict(self, data, index):

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
        fname = index.split('/')[-1]
        csv_filename = os.path.join(self.save_path, timestamp_filename(fname, '.csv'))
        with open(csv_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(columns) 

            query_count = 0
            for idx, query in data.queries.items(): 
                rank = 'NA'
                matching_text = 'NA'
                score = 'NA'
                top_result_match = False
                in_top_10 = False
                print(query_count, query)
                expected_id = data.relations[idx]
                doc_texts, doc_ids, doc_scores = self.retriever.retrieve_topn(query)
                if index != 'msmarco_index':
                    doc_ids = ['.'.join(i.split('.')[:-1]) for i in doc_ids]
                    print("found ids: ", doc_ids)
                    print("expected id: ", expected_id)

                if expected_id in doc_ids:
                    print("FOUND EXPECTED ID")
                    in_top_10 = True
                    rank = doc_ids.index(expected_id)
                    matching_text = data.collection[expected_id]
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
        
    def eval(self, data, index):
        
        df = self.predict(data, index)

        num_queries = df['queries'].nunique()

        proportion_expected_in_top_10 = 0
        proportion_expected_is_top = 0

        if True in df['in_top_10'].unique():
            proportion_expected_in_top_10 = np.round(df['in_top_10'].value_counts(normalize = True)[True], 2)
        if True in df['top_result_match'].unique():
            proportion_expected_is_top = np.round(df['top_result_match'].value_counts(normalize = True)[True], 2)
        
        agg_results = {
            "num_queries": num_queries,
            "proportion_expected_in_top_10": proportion_expected_in_top_10,
            "proportion_expected_is_top": proportion_expected_is_top
        }
        return agg_results

class MSMarcoRetrieverEvaluator(RetrieverEvaluator):

    def __init__(
            self, 
            model_name=EmbedderConfig.MODEL_ARGS['model_name'],
            transformer_path=LOCAL_TRANSFORMERS_DIR,
            index='msmarco_index',
            save_path=SAVE_PATH, 
            encoder_args=EmbedderConfig.MODEL_ARGS, 
            similarity_args=SimilarityConfig.MODEL_ARGS,
            use_gpu=False,
            new_model=False
        ):

        super().__init__(model_name, transformer_path, save_path)

        self.data = MSMarcoData()
        self.index_path = os.path.join(save_path, index)
        if not os.path.exists(self.index_path):  
            print("Making new embeddings index at {}".format(str(self.index_path)))
            os.makedirs(self.index_path)
            self.encoder = SentenceEncoder(encoder_args, self.index_path, use_gpu)
            self.make_index(encoder=self.encoder, corpus_path=None)
        self.retriever = SentenceSearcher(self.index_path, transformer_path, encoder_args, similarity_args)
        if new_model == True:
            self.results = self.eval(data=self.data, index=index)

class IndomainRetrieverEvaluator(RetrieverEvaluator):

    def __init__(
            self, 
            model_name=EmbedderConfig.MODEL_ARGS['model_name'],
            transformer_path=LOCAL_TRANSFORMERS_DIR,
            index=SENT_INDEX_PATH,
            save_path=SAVE_PATH, 
            corpus_path='/Users/katherinedowdy/Documents/GameChanger/gamechanger-ml/gamechangerml/data/corpus_test', # default corpus path??
            encoder_args=EmbedderConfig.MODEL_ARGS, 
            similarity_args=SimilarityConfig.MODEL_ARGS,
            use_gpu=False,
            new_model=False
        ):

        super().__init__(model_name, transformer_path, save_path)

        self.data = RetrieverDomainData()
        self.index_path = index
        if not os.path.exists(self.index_path):  
            print("Making new embeddings index at {}".format(str(self.index_path)))
            os.makedirs(self.index_path)
            self.encoder = SentenceEncoder(encoder_args, self.index_path, use_gpu)
            self.make_index(encoder=self.encoder, corpus_path=corpus_path)
        self.retriever = SentenceSearcher(self.index_path, transformer_path, encoder_args, similarity_args)
        if new_model == True:
            self.results = self.eval(data=self.data, index=index)

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
            self.results = self.eval_nli()

    def predict_nli(self):
        '''Get rank predictions from similarity model'''

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
        '''Get summary stats of predicted vs. expected ranking for NLI'''

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
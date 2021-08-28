import pandas as pd
import numpy as np
from gamechangerml.src.utilities.model_helper import *
from gamechangerml.configs.config import ValidationConfig
from gamechangerml.api.utils.logger import logger

class ValidationData():

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        self.validation_dir = validation_config['validation_dir']

class SQuADData(ValidationData):

    def __init__(self, sample_limit=None, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.dev = open_json(validation_config['squad']['dev'], self.validation_dir)
        self.queries = self.get_squad_sample(sample_limit)
    
    def get_squad_sample(self, sample_limit):
        '''Format SQuAD data into list of dictionaries (length = sample size)'''

        data_limit = len(self.dev['data'])

        if sample_limit:
            data_limit = np.min([data_limit, sample_limit])
            par_limit = sample_limit // data_limit
        else:
            par_limit = np.max([len(d['paragraphs']) for d in self.dev['data']])

        count = 0
        queries = []
        for p in range(par_limit):
            for d in range(data_limit):
                try:
                    base = self.dev['data'][d]['paragraphs'][p]
                    context = base['context']
                    questions = base['qas']
                    q_limit = np.min([2, par_limit, len(questions)])
                    for q in range(q_limit):
                        if count < sample_limit:
                            count += 1
                            mydict = {
                                "search_context": context,
                                "question": questions[q]['question'],
                                "id": questions[q]['id'],
                                "null_expected": questions[q]['is_impossible'],
                                "expected": questions[q]['answers']
                            }
                            queries.append(mydict)
                        else:
                            break
                except:
                    pass
        
        logger.info("Generated {} question/answer pairs from SQuAD dataset".format(len(queries)))

        return queries

class QADomainData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.all_queries = open_json(validation_config['question_gc']['queries'], self.validation_dir)
        self.queries = self.check_queries()

    def check_queries(self):
        '''Check that in-domain examples contain expected answers in their context'''

        checked = []
        for test in self.all_queries['test_queries']:
            alltext = normalize_answer(' '.join(test['search_context']))
            checked_answers = [i for i in test['expected'] if normalize_answer(i['text']) in alltext]
            test['expected'] = checked_answers
            if test['expected'] != []:
                checked.append(test)
            else:
                logger.info("Could not add {} to test queries: answer not in context".format(test['question']))
        
        logger.info("Generated {} question/answer pairs from in-domain data".format(len(checked)))

        return checked

class MSMarcoData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.queries = open_json(validation_config['msmarco']['queries'], self.validation_dir)
        self.collection = open_json(validation_config['msmarco']['collection'], self.validation_dir)
        self.relations = open_json(validation_config['msmarco']['relations'], self.validation_dir)
        self.metadata = open_json(validation_config['msmarco']['metadata'], self.validation_dir)
        self.corpus = self.get_msmarco_corpus()

    def get_msmarco_corpus(self):
        '''Format MSMarco so it can be indexed like the GC corpus'''

        return [(x, y, '') for x, y in self.collection.items()]

class RetrieverGSData(ValidationData):

    def __init__(self, available_ids, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.samples = pd.read_csv(os.path.join(self.validation_dir, validation_config['retriever_gc']['gold_standard']), names=['query', 'document'])
        self.queries, self.collection, self.relations = self.dictify_data(available_ids)
    
    def dictify_data(self, available_ids):
        '''
        Filter out any validation queries whose documents areen't in the index. 
        Forrmat gold standard csv examples into MSMarco format.
        '''

        ids = ['.'.join(i.strip('\n').split('.')[:-1]).strip().lstrip() for i in available_ids]


        self.samples['document'] = self.samples['document'].apply(lambda x: [i.strip().lstrip() for i in x.split(';')])
        self.samples = self.samples.explode('document')
        df = self.samples[self.samples['document'].isin(ids)] # check ids are in the index
        if df.shape[0] < self.samples.shape[0]:
            all_ids = self.samples['document'].unique()
            missing_ids = [i for i in all_ids if i not in ids]
            logger.info("Validation IDs not in the index (removed from validation set): {}".format(missing_ids))

        df = df.groupby('query').agg({'document': lambda x: x.tolist()}).reset_index()
        query_list = df['query'].to_list()
        doc_list = df['document'].to_list()
        q_idx = ["query_" + str(i) for i in range(len(query_list))]
        queries = dict(zip(q_idx, query_list))
        collection = dict(zip(all_ids, all_ids))
        relations = dict(zip(q_idx, doc_list))

        logger.info("Generated {} test queries of gold standard data".format(len(query_list)))

        return queries, collection, relations

class NLIData(ValidationData):

    def __init__(self, sample_limit, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.matched = open_jsonl(validation_config['nli']['matched'], self.validation_dir)
        self.mismatched = open_jsonl(validation_config['nli']['mismatched'], self.validation_dir)
        self.sample_csv = self.get_sample_csv(sample_limit)
        self.query_lookup = dict(zip(self.sample_csv['promptID'], self.sample_csv['sentence1']))

    def get_sample_csv(self, sample_limit):
        '''Format NLI data into smaller sample for evaluation'''

        match_df = pd.DataFrame(self.matched)
        mismatched_df = pd.DataFrame(self.mismatched)
        match_df['set'] = 'matched'
        mismatched_df['set'] = 'mismatched'
        both = pd.concat([match_df, mismatched_df])
        # assign int ranks based on gold label
        gold_labels_map = {
            'entailment': 2,
            'neutral': 1, 
            'contradiction': 5
        }
        both['gold_label_int'] = both['gold_label'].map(gold_labels_map)

        # filter out propmtIDs that don't have a clear 0, 1, 2 rank
        sum_map = both.groupby('promptID')['gold_label_int'].sum().to_dict()
        both['rank_sum'] = both['promptID'].map(sum_map)
        both = both[both['rank_sum']==8]

        # map ranks
        rank_map =  {
            'entailment': 0,
            'neutral': 1, 
            'contradiction': 2
        }
        both['expected_rank'] = both['gold_label'].map(rank_map)

        cats = both['genre'].nunique()

        # get smaller sample df with even proportion of genres across matched/mismatched
        sample = pd.DataFrame()
        for i in both['genre'].unique():
            subset = both[both['genre']==i].sort_values(by='promptID')
            if sample_limit:
                split = sample_limit * 3 // cats
                subset = subset.head(split)
            sample = pd.concat([sample, subset])

        logger.info(("Created {} sample sentence pairs from {} unique queries:".format(sample.shape[0], sample_limit)))

        return sample[['genre', 'gold_label', 'pairID', 'promptID', 'sentence1', 'sentence2', 'expected_rank']]

class MatamoFeedback():
    
    def __init__(self, matamo_feedback_path):
        
        self.matamo = pd.read_csv(matamo_feedback_path)
        self.intel, self.qa = self.split_matamo()
    
    def split_matamo(self):
        '''Split QA queries from intelligent search queries'''
        
        df = self.matamo
        df['source'] = 'matamo'
        df['correct'] = df['event_name'].apply(lambda x: ' '.join(x.split('_')[-2:])).map({'thumbs up': True, 'thumbs down': False})
        df['type'] = df['event_name'].apply(lambda x: ' '.join(x.split('_')[:-2]))
        intel = df[df['type']=='intelligent search'].copy()
        qa = df[df['type']=='qa'].copy()
    
        def process_matamo(df):
            '''Reformat Matamo feedback'''

            queries = []
            cols = [i for i in df.columns if i[:5]=='value']

            def process_row(row, col_name):
                '''Split the pre-colon text from rows'''

                if ':' in row:
                    row = row.split(':')
                    key = row[0]
                    vals = ':'.join(row[1:])
                    return key, vals
                else:
                    return col_name, row

            for i in df.index:
                query = {}
                query['date'] = df.loc[i, 'createdAt']
                query['source'] = 'matamo'
                query['correct_match'] = df.loc[i, 'correct']
                for j in cols:
                    row = df.loc[i, j]
                    key, val = process_row(row, j)
                    query[key] = val
                    if key in ['question', 'search_text', 'QA answer']:
                        clean_val = normalize_answer(val)
                        clean_key = key + '_clean'
                        query[clean_key] = clean_val
                queries.append(query)

            return pd.DataFrame(queries)

        return process_matamo(intel), process_matamo(qa)

class SearchHistory():
    
    def __init__(self, search_history_path):
    
        self.history = pd.read_csv(search_history_path)
        self.intel = self.split_feedback()
        
    def split_feedback(self):
        
        df = self.history
        
        def clean_quot(string):
            return string.replace('&quot;', "'").replace("&#039;", "'").lower()
        
        def clean_doc(string):
            return string.strip('.pdf')

        def is_question(string):
            '''If we find a good way to use search history for QA validation (not used currently)'''

            question_words = ['what', 'who', 'where', 'why', 'how', 'when']
            if '?' in string:
                return True
            else:
                return bool(set(string.lower().split()).intersection(question_words))
        
        df['source'] = 'user_history'
        df['correct_match'] = True
        #df['is_question'] = sh['search'].apply(lambda x: is_question(x))
        df.rename(columns = {'documenttime': 'date', 'search': 'search_text', 'document': 'title_returned'}, inplace = True)
        df['title_returned'] = df['title_returned'].apply(lambda x: clean_doc(x))
        df['search_text'] = df['search_text'].apply(lambda x: clean_quot(x))
        df['search_text_clean'] = df['search_text'].apply(lambda x: normalize_answer(x))
        df.drop(columns = ['idvisit', 'idaction_name', 'search_cat', 'searchtime'], inplace = True)
        
        return df
    
class SearchValidationData(ValidationData):
    
    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):
        
        ##TODO: option to add new data to existing formatted data/add only new records
        super().__init__(validation_config)
        self.matamo_path = os.path.join(self.validation_dir, validation_config['matamo_feedback_file'])
        self.history_path = os.path.join(self.validation_dir, validation_config['search_history_file'])
        self.matamo_data = MatamoFeedback(self.matamo_path)
        self.history_data = SearchHistory(self.history_path)
        self.intel_search = pd.concat([self.matamo_data.intel, self.history_data.intel]).reset_index()
        self.qa_search = self.matamo_data.qa
        
class QASearchData(SearchValidationData):
    
    ##TODO: add context relations attr for QASearchData
    
    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):
        
        super().__init__(validation_config)
        self.queries, self.collection, self.meta_relations, self.relations = self.make_qa()
        
    def make_qa(self):
        
        qa = self.qa_search
        
        # get set of queries + make unique query dict
        qa_queries = set(qa['question_clean'])
        qa_search_queries = update_dictionary(old_dict = {}, new_additions = qa_queries, prefix = 'Q')
        
        # get set of docs + make unique doc dict
        qa_answers = set(qa['QA answer_clean'])
        qa_search_results = update_dictionary(old_dict = {}, new_additions = qa_answers, prefix = 'A')
        
        # map IDs back to df
        qa = map_ids(qa_search_queries, qa, 'question_clean', 'key')
        qa = map_ids(qa_search_results, qa, 'QA answer_clean', 'value')
        
        # create new QA metadata rels
        qa_metadata = {} # TODO: add option to add existing metadata
        new_qa_metadata = update_meta_relations(qa_metadata, qa, 'question', 'QA answer')
        
        # filtere the metadata to only get relations we want to test against
        qa_rels = filter_rels(new_qa_metadata, min_matches=0)
        
        return qa_search_queries, qa_search_results, new_qa_metadata, qa_rels


class IntelSearchData(SearchValidationData):
    
    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):
        
        super().__init__(validation_config)
        self.queries, self.collection, self.meta_relations = self.make_intel()
        
    def make_intel(self):
        
        intel = self.intel_search
        
        int_queries = set(intel['search_text_clean'])
        intel_search_queries = update_dictionary(old_dict = {}, new_additions = int_queries, prefix ='S')
        
        int_docs = set(intel['title_returned'])
        intel_search_results = update_dictionary(old_dict = {}, new_additions = int_docs, prefix ='R')
        
        # map IDS back to dfs
        intel = map_ids(intel_search_queries, intel, 'search_text_clean', 'key')
        intel = map_ids(intel_search_results, intel, 'title_returned', 'value')

        # create new intel search metadata rels
        intel_metadata = {} # TODO: add option to add existing metadata
        new_intel_metadata = update_meta_relations(intel_metadata, intel, 'search_text', 'title_returned')
        
        # filtere the metadata to only get relations we want to test against
        #intel_rels = filter_rels(new_intel_metadata, min_matches=2)
        
        return intel_search_queries, intel_search_results, new_intel_metadata
    
    def filter_rels(self, min_correct_matches):
        '''Filter relations by criteria'''
        
        correct_rels = {}
        incorrect_rels = {}
        for key in self.meta_relations:
            acceptable_positive_results = []
            negative_results = []
            for match in self.meta_relations[key]:
                result = self.meta_relations[key][match]
                sources = [i['source'] for i in result['exact_matches']]
                if result['correct_match'] == True:
                    if 'matamo' in sources: # we trust matamo data
                        acceptable_positive_results.append(match)
                    elif result['times_matched'] >= min_correct_matches: # only pull history matches occurring more than x times
                        acceptable_positive_results.append(match)
                elif result['correct_match'] == False:
                    negative_results.append(match)

            if acceptable_positive_results != []:
                correct_rels[key] = acceptable_positive_results
            if negative_results != []:
                incorrect_rels[key] = negative_results
            
        return correct_rels, incorrect_rels

class QEXPDomainData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.data = open_json(validation_config['qe_gc'], self.validation_dir)['queries']

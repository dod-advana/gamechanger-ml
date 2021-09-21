import pandas as pd
import numpy as np
from gamechangerml.src.utilities.text_utils import normalize_answer, get_tokens
from gamechangerml.src.utilities.test_utils import *
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
        Filter out any validation queries whose documents aren't in the index. 
        Format gold standard csv examples into MSMarco format.
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

class QEXPDomainData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.data = open_json(validation_config['qe_gc'], self.validation_dir)['queries']
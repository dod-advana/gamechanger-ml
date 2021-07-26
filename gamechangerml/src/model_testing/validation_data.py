import pandas as pd
from gamechangerml.src.utilities.model_helper import *
from gamechangerml.configs.config import ValidationConfig

class ValidationData():

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        self.validation_dir = validation_config['validation_dir']

class SQuADData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.dev = open_json(validation_config['squad']['dev'], self.validation_dir)
        self.sample_limit = validation_config['squad']['sample_limit']
        self.queries = self.get_squad_sample()
    
    def get_squad_sample(self):
        '''Format SQuAD data into list of dictionaries (length = sample size)'''

        queries = []
        count = 0
        for entry in self.dev['data']:
            for para in entry['paragraphs']:
                context = para['context']
                if type(context) == str:
                    context = [context]
                for test in para['qas']:
                    if count < self.sample_limit:
                        count += 1
                        mydict = {
                            "search_context": context,
                            "question": test['question'],
                            "id": test['id'],
                            "null_expected": test['is_impossible'],
                            "expected": test['answers']
                        }
                        queries.append(mydict)
                    break
        
        print("Generated {} SQuAD queries".format(len(queries)))

        return queries

class QADomainData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.queries = open_json(validation_config['question_gc']['queries'], self.validation_dir)
        self.checked_queries = self.check_queries()

    def check_queries(self):
        '''Check that in-domain examples contain expected answers in their context'''

        checked = []
        for test in self.queries['test_queries']:
            alltext = normalize_answer(' '.join(test['search_context']))
            checked_answers = [i for i in test['expected'] if normalize_answer(i['text']) in alltext]
            test['expected'] = checked_answers
            if test['expected'] != []:
                checked.append(test)
            else:
                print("Could not add {} to test queries: answer not in context".format(test['question']))
        
        print("Number of in-domain question/answer examples: {}".format(len(checked)))

        return checked

class MSMarcoData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.orig_queries = open_json(validation_config['msmarco']['queries'], self.validation_dir)
        self.collection = open_json(validation_config['msmarco']['collection'], self.validation_dir)
        self.relations = open_json(validation_config['msmarco']['relations'], self.validation_dir)
        self.metadata = open_json(validation_config['msmarco']['metadata'], self.validation_dir)
        self.queries = self.filter_queries()
        self.corpus = self.get_msmarco_corpus()

    def get_msmarco_corpus(self):
        '''Format MSMarco so it can be indexed like the GC corpus'''

        return [(x, y, '') for x, y in self.collection.items()]

    def filter_queries(self):
        '''Filter out MSMarco examples with more than one top expected doc'''

        include = [i for i, x in self.relations.items() if len(x)==1]
        print("Number MSMarco test queries: ", len(include))

        return {x: self.orig_queries[x] for x in include}

class RetrieverDomainData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        self.samples = pd.read_csv(os.path.join(self.validation_dir, validation_config['retriever_gc']['gold_standard']), names=['query', 'document'])
        self.queries, self.collection, self.relations = self.dictify_data()
    
    def dictify_data(self):
        '''Format gold standard csv examples into MSMarco format'''

        self.samples['document'] = self.samples['document'].apply(lambda x: x.split(';'))
        self.samples = self.samples.explode('document')
        query_list = self.samples['query'].to_list()
        doc_list = self.samples['document'].to_list()
        q_idx = ["query_" + str(i) for i in range(len(query_list))]
        d_idx = ["doc_" + str(i) for i in range(len(doc_list))]
        queries = dict(zip(q_idx, query_list))
        collection = dict(zip(d_idx, doc_list))
        relations = dict(zip(q_idx, d_idx))

        return queries, collection, relations

class NLIData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.matched = open_jsonl(validation_config['nli']['matched'], self.validation_dir)
        self.mismatched = open_jsonl(validation_config['nli']['mismatched'], self.validation_dir)
        self.sample_csv = self.get_sample_csv()
        self.query_lookup = dict(zip(self.sample_csv['promptID'], self.sample_csv['sentence1']))

    def get_sample_csv(self):
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

        # get smaller sample df with even proportion of genres across matched/mismatched
        sample = pd.DataFrame()
        for i in both['genre'].unique():
            subset = both[both['genre']==i].sort_values(by='promptID').head(300)
            sample = pd.concat([sample, subset])

        print("Created {} sample sentence pairs:".format(sample.shape[0]))
        print(sample.head())

        return sample[['genre', 'gold_label', 'pairID', 'promptID', 'sentence1', 'sentence2', 'expected_rank']]
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
        #self.train = open_json(validation_config['squad']['train'], self.validation_dir)
        self.queries = self.get_squad_sample()
    
    def get_squad_sample(self):

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

class MSMarcoData(ValidationData):

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.queries = open_json(validation_config['msmarco']['queries'], self.validation_dir)
        self.collection = open_json(validation_config['msmarco']['collection'], self.validation_dir)
        self.relations = open_json(validation_config['msmarco']['relations'], self.validation_dir)
        self.metadata = open_json(validation_config['msmarco']['metadata'], self.validation_dir)
        self.msmarco_corpus = self.get_msmarco_corpus()

    def get_msmarco_corpus(self):

        return [(x, y, '') for x, y in self.collection.items()]
    

class NLIData(ValidationData):
    '''
    Formats the raw NLI data into evaluation format for similarity model.
    '''
    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        super().__init__(validation_config)
        self.matched = open_jsonl(validation_config['nli']['matched'], self.validation_dir)
        self.mismatched = open_jsonl(validation_config['nli']['mismatched'], self.validation_dir)
        self.sample_csv = self.get_sample_csv()

    def get_sample_csv(self):
        '''
        From the paper: 'All of the genres appear in the test and development sets, but only five are included 
        in the training set. Models thus can be evaluated on both the matched test examples, which are derived 
        from the same sources as those in the training set, and on the mismatched examples, which do not 
        closely resemble any of those seen at training time.'
        '''

        match_df = pd.DataFrame(self.matched)
        mismatched_df = pd.DataFrame(self.mismatched)
        match_df['set'] = 'matched'
        mismatched_df['set'] = 'mismatched'
        both = pd.concat[match_df, mismatched_df]
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

        return sample[['genre', 'gold_label', 'pairID', 'promptID', 'sentence1', 'sentence2', 'expected_rank']]
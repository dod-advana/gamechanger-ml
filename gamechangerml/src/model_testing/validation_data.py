from gamechangerml.src.utilities.model_helper import *
from gamechangerml.configs.config import ValidationConfig

class ValidationData():

    def __init__(self, validation_config=ValidationConfig.DATA_ARGS):

        self.validation_dir = validation_config['validation_dir']

class SQuADData(ValidationData):

    def __init__(self, validation_config):

        super().__init__(validation_config)
        self.dev = open_json(validation_config['squad']['dev'], self.validation_dir)
        self.train = open_json(validation_config['squad']['train'], self.validation_dir)

class NLIData(ValidationData):

    def __init__(self, validation_config):

        super().__init__(validation_config)
        self.matched = open_json(validation_config['nli']['matched'], self.validation_dir)
        self.mismatched = open_json(validation_config['nli']['mismatched'], self.validation_dir)
        self.train = open_json(validation_config['nli']['train'], self.validation_dir)

class MSMarcoData(ValidationData):

    def __init__(self, validation_config):

        super().__init__(validation_config)
        self.queries = open_json(validation_config['msmarco']['queries'], self.validation_dir)
        self.collection = open_json(validation_config['msmarco']['collection'], self.validation_dir)
        self.relations = open_json(validation_config['msmarco']['relations'], self.validation_dir)
        self.metadata = open_json(validation_config['msmarco']['metadata'], self.validation_dir)


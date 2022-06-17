import os
from gamechangerml.configs.config import DefaultConfig

# See transformers.RobertaTokenizer.from_pretrained()
BASE_MODEL_NAME = "distilroberta-base"

# Maximum number of tokens to chunk text into when applying section labels
# with the section parsing model.
MAX_TOKENS = 500

# See transformers.pipelines.token_classification.
# Reference: https://huggingface.co/transformers/v4.10.1/_modules/transformers/pipelines/token_classification.html
AGGREGATION_STRATEGY = "simple"

# Path to the pretrained section parsing model.
# TODO: change to not be local path. download from S3 ?
MODEL_PATH = "/Users/oliviakroop/Desktop/Projects/GameChanger/GitHub_GameChanger/gamechanger-ml/gamechangerml/src/section_classifier/model/checkpoint-1190"

# Minimum score (assigned by the label parsing model) for a References section
# body to be included
REFERENCES_MIN_SCORE = 0.8

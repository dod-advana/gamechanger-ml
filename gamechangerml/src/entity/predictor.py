import logging

from transformers import pipeline

import gamechangerml.src.entity.version as v

logger = logging.getLogger(__name__)


class Predictor(object):
    __version__ = v.__version__

    def __init__(self, model_name_or_path):
        """
        Wrapper for HF `pipeline`

        Args:
            model_name_or_path (str): directory of a trained model or a
                HF named model

        Raises:
             ValueError if `model_name_or_path` is missing
             OSError if the model cannot be loaded
        """
        if not model_name_or_path:
            raise ValueError("no value for `model_name_or_path`")
        try:
            self.ner_pipe = pipeline(
                "ner", model=model_name_or_path, grouped_entities=True
            )
        except OSError as e:
            raise e

    def __call__(self, seq):
        """
        Calls the pipeline with `seq`

        Args:
            seq (str): sequence to predict

        Returns:
            List[dict]: prediction results, e.g.,

            [
             {'entity_group': 'GCORG', 'score': 0.9965277761220932,
              'word': 'defense threat reduction agency', 'start': 4, 'end': 35},
             {'entity_group': 'GCORG', 'score': 0.9997164011001587,
              'word': 'dt', 'start': 37, 'end': 39},
            {'entity_group': 'GCPER', 'score': 0.9998179872830709,
              'word': 'secretary of state', 'start': 52, 'end': 70}]
        """
        return self.ner_pipe(seq)

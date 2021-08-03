import logging

from transformers import pipeline

import gamechangerml.src.entity.version as v

logger = logging.getLogger(__name__)


class PredictNER:
    """
    Wrapper for HF NER `pipeline`

    Args:
        model_name_or_path (str): directory of a trained model or an
            HF named model

    Raises:
         ValueError if `model_name_or_path` is missing or cannot be loaded
    """

    __version__ = v.__version__

    def __init__(self, model_name_or_path=None):

        logger.info(
            "{} version {}".format(self.__class__.__name__, self.__version__)
        )

        if model_name_or_path is not None:
            raise ValueError("no value for `model_name_or_path`")
        try:
            self.ner_pipe = pipeline(
                "ner", model=model_name_or_path, grouped_entities=True
            )
        except OSError as e:
            msg = "`model_name_or_path` could not be found or loaded; got "
            msg += "{}\n".format(model_name_or_path)
            msg += "exception type / string {}: {}".format(type(e), str(e))
            raise ValueError(msg)

        self.empty_list = list()

    def __call__(self, seq):
        """
        Calls the pipeline with `seq`

        Args:
            seq (str): sequence to predict

        Returns:
            List[dict]: prediction results

        Example:
           >>> ner_pipe = PredictNER("model/directory")
           >>> seq = "The Defense Threat Reduction Agency (DTRA), and the Secretary of State"
           >>> ner_pipe(seq)
           [
             {'entity_group': 'GCORG',
              'score': 0.9965277761220932,
              'word': 'defense threat reduction agency',
              'start': 4,
              'end': 35},
             {'entity_group': 'GCORG',
              'score': 0.9997164011001587,
              'word': 'dt',
              'start': 37,
              'end': 39},
             {'entity_group': 'GCPER',
              'score': 0.9998179872830709,
              'word': 'secretary of state',
              'start': 52,
              'end': 70}
           ]
        """
        if not seq:
            return self.empty_list
        else:
            return self.ner_pipe(seq)

from os.path import join
from .utils import LOCAL_TRANSFORMERS_DIR
from .qa_evaluator import QAEvaluator
from ..validation_data import SQuADData
from gamechangerml.src.utilities import create_directory_if_not_exists

class SQuADQAEvaluator(QAEvaluator):
    def __init__(
        self,
        model_name,
        qa_type,
        nbest,
        null_threshold,
        model=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        use_gpu=False,
        sample_limit=None,
        data_name="squad",
    ):

        super().__init__(
            model_name,
            qa_type,
            nbest,
            null_threshold,
            model,
            transformer_path,
            use_gpu,
            data_name,
        )

        self.data = SQuADData(sample_limit)
        self.eval_path = create_directory_if_not_exists(join(self.model_path, "evals_squad"))
        self.results = self.eval(data=self.data, eval_path=self.eval_path)

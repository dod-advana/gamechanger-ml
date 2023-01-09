import os
from .utils import LOCAL_TRANSFORMERS_DIR
from .qa_evaluator import QAEvaluator
from ..validation_data import QADomainData
from gamechangerml.src.utilities import create_directory_if_not_exists


class IndomainQAEvaluator(QAEvaluator):
    def __init__(
        self,
        model_name,
        qa_type,
        nbest,
        null_threshold,
        model=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        use_gpu=False,
        data_name="domain",
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

        self.data = QADomainData()
        self.eval_path = create_directory_if_not_exists(
            os.path.join(self.model_path, "evals_gc")
        )
        self.results = self.eval(data=self.data, eval_path=self.eval_path)

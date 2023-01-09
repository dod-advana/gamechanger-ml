import os
import numpy as np
from .utils import LOCAL_TRANSFORMERS_DIR, logger
from .similarity_evaluator import SimilarityEvaluator
from ..validation_data import NLIData
from gamechangerml.src.utilities import create_directory_if_not_exists

class NLIEvaluator(SimilarityEvaluator):
    def __init__(
        self,
        sim_model_name,
        model=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        sample_limit=None,
        use_gpu=False,
    ):

        super().__init__(sim_model_name, model, transformer_path, use_gpu)

        self.data = NLIData(sample_limit)
        self.eval_path = create_directory_if_not_exists(os.path.join(self.model_path, "evals_nli"))
        self.results = self.eval(
            predictions=self.predict_nli(), eval_path=self.eval_path
        )

    def predict_nli(self):
        """Get rank predictions from similarity model"""

        df = self.data.sample_csv
        ranks = {}
        count = 0
        for i in df["promptID"].unique():
            subset = df[df["promptID"] == i]
            iddict = dict(zip(subset["sentence2"], subset["pairID"]))
            texts = [i for i in iddict.keys()]
            ids = [i for i in iddict.values()]
            query = self.data.query_lookup[i]
            logger.info("S-{}: {}".format(count, query))
            rank = 0
            for result in self.model.re_rank(query, texts, ids):
                match_id = result["id"]
                ranks[match_id] = rank
                rank += 1

            count += 1

        df["predicted_rank"] = df["pairID"].map(ranks)
        df.dropna(subset=["predicted_rank"], inplace=True)
        df["predicted_rank"] = df["predicted_rank"].map(int)
        df["match"] = np.where(df["predicted_rank"] == df["expected_rank"], 1, 0)

        return df

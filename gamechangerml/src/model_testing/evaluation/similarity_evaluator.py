import os
import numpy as np
from datetime import datetime
from .utils import LOCAL_TRANSFORMERS_DIR, logger
from .transformer_evaluator import TransformerEvaluator
from ..metrics import get_MRR
from gamechangerml.src.search.sent_transformer.model import SimilarityRanker
from gamechangerml.src.utilities import save_json
from gamechangerml.src.utilities.test_utils import timestamp_filename, get_user, clean_nans



class SimilarityEvaluator(TransformerEvaluator):
    def __init__(
        self,
        sim_model_name,
        model=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        use_gpu=False,
    ):

        super().__init__(transformer_path, use_gpu)

        if model:
            self.model = model
        else:
            self.model = SimilarityRanker(sim_model_name, transformer_path)
        self.sim_model_name = sim_model_name
        self.model_path = os.path.join(transformer_path, sim_model_name)

    def eval(self, predictions, eval_path):
        """Get summary stats of predicted vs. expected ranking for NLI"""

        df = predictions
        csv_filename = os.path.join(eval_path, timestamp_filename("nli_eval", ".csv"))
        df.to_csv(csv_filename)

        # get overall stats
        all_accuracy = np.round(df["match"].mean(), 2)
        top_accuracy = np.round(df[df["expected_rank"] == 0]["match"].mean(), 2)

        # get MRR
        top_only = df[
            df["expected_rank"] == 0
        ].copy()  # take only the expected top results
        top_only["reciprocal_rank"] = top_only["predicted_rank"].apply(
            lambda x: 1 / (x + 1)
        )  # add one because ranks are 0-indexed
        _mrr = get_MRR(list(top_only["reciprocal_rank"]))

        num_queries = df["promptID"].nunique()
        num_sentence_pairs = df.shape[0]

        user = get_user(logger)

        agg_results = {
            "user": user,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.sim_model_name,
            "validation_data": "NLI",
            "query_count": clean_nans(num_queries),
            "pairs_count": clean_nans(num_sentence_pairs),
            "all_accuracy": clean_nans(all_accuracy),
            "top_accuracy": clean_nans(top_accuracy),
            "MRR": _mrr,
        }

        output_file = timestamp_filename("sim_model_eval", ".json")
        save_json(output_file, eval_path, agg_results)
        logger.info(f"Saved evaluation to {str(os.path.join(eval_path, output_file))}")

        return agg_results


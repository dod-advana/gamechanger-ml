import os
import math
import csv
import signal
import pandas as pd
import numpy as np
from datetime import datetime
from .transformer_evaluator import TransformerEvaluator
from .utils import LOCAL_TRANSFORMERS_DIR, logger
from ..metrics import compute_QA_f1, get_precision, get_recall, get_f1
from gamechangerml.src.utilities import save_json, TimeoutException
from gamechangerml.src.search.QA.QAReader import DocumentReader as QAReader
from gamechangerml.src.utilities.text_utils import normalize_answer
from gamechangerml.src.utilities.test_utils import timestamp_filename, get_user


class QAEvaluator(TransformerEvaluator):
    def __init__(
        self,
        model_name,
        qa_type,
        nbest,
        null_threshold,
        model=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        use_gpu=False,
        data_name=None,
    ):

        super().__init__(transformer_path, use_gpu)

        self.model_name = model_name
        self.model_path = os.path.join(transformer_path, model_name)
        logger.info(f"model path: {str(self.model_path)}")
        if not os.path.exists(self.model_path):
            logger.warning("Model directory provided does not exist.")
        if model:
            self.model = model
        else:
            self.model = QAReader(
                transformer_path,
                model_name,
                qa_type,
                nbest,
                null_threshold,
                use_gpu,
            )
        self.data_name = data_name

    def compare(self, prediction, query):
        """Compare predicted to expected answers"""

        exact_match = 0
        partial_match = 0  # true positive
        true_negative = 0
        false_negative = 0
        false_positive = 0
        best_partial_f1 = 0

        if prediction["text"] == "":
            if query["null_expected"] == True:
                exact_match = partial_match = true_negative = 1
            else:
                false_negative = 1
        elif query["null_expected"] == True:
            false_positive = 1
        else:
            clean_pred = normalize_answer(prediction["text"])
            clean_answers = set(
                [normalize_answer(i["text"]) for i in query["expected"]]
            )
            if clean_pred in clean_answers:
                exact_match = partial_match = best_partial_f1 = 1
            else:
                partial_f1 = []
                for i in clean_answers:
                    f1_score = compute_QA_f1(clean_pred, i)
                    partial_f1.append(f1_score)
                best_partial_f1 = max(partial_f1)
            partial_match = math.ceil(best_partial_f1)  # return 0 or 1
            false_positive = 1 - partial_match

        return (
            exact_match,
            partial_match,
            true_negative,
            false_negative,
            false_positive,
            best_partial_f1,
        )

    def predict(self, data, eval_path):
        """Get answer predictions"""

        columns = [
            "index",
            "queries",
            "actual_answers",
            "predicted_answer",
            "exact_match",
            "partial_match",
            "best_partial_f1",
            "true_negative",
            "false_negative",
            "false_positive",
        ]

        query_count = 0

        csv_filename = os.path.join(
            eval_path, timestamp_filename(self.data_name, ".csv")
        )
        with open(csv_filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(columns)

            for query in data.queries:
                signal.alarm(20)
                try:
                    logger.info(
                        "Q-{}: {}".format(query_count, query["question"])
                    )
                    actual = query["expected"]
                    context = query["search_context"]
                    if type(context) == str:
                        context = [context]
                    prediction = self.model.answer(query["question"], context)[
                        0
                    ]
                    (
                        exact_match,
                        partial_match,
                        true_negative,
                        false_negative,
                        false_positive,
                        best_partial_f1,
                    ) = self.compare(prediction, query)

                    row = [
                        [
                            str(query_count),
                            str(query["question"]),
                            str(actual),
                            str(prediction),
                            str(exact_match),
                            str(partial_match),
                            str(best_partial_f1),
                            str(true_negative),
                            str(false_negative),
                            str(false_positive),
                        ]
                    ]
                    csvwriter.writerows(row)
                    query_count += 1
                except TimeoutException:
                    logger.info("Query timed out before answer")
                    query_count += 1
                    continue
                else:
                    signal.alarm(0)

        return pd.read_csv(csv_filename)

    def eval(self, data, eval_path):
        """Get evaluation stats across predicted/expected answer comparisons"""

        df = self.predict(data, eval_path)

        num_queries = df["queries"].nunique()
        if num_queries > 0:
            exact_match = np.round(np.mean(df["exact_match"].to_list()), 2)
            true_positives = df["partial_match"].map(int).sum()
            false_positives = df["false_positive"].map(int).sum()
            false_negatives = df["false_negative"].map(int).sum()
            precision = get_precision(true_positives, false_positives)
            recall = get_recall(true_positives, false_negatives)
            f1 = get_f1(precision, recall)
            average_f1 = np.round(
                np.mean(df["best_partial_f1"].map(float).to_list()), 3
            )
        else:
            exact_match = precision = recall = f1 = average_f1 = 0
        user = get_user(logger)

        agg_results = {
            "user": user,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "validation_data": self.data_name,
            "query_count": num_queries,
            "exact_match_accuracy": exact_match,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_f1": average_f1,  # degree of matching-ness of answers, value from 0-1
        }

        file = "_".join(["qa_eval", self.data_name])
        output_file = timestamp_filename(file, ".json")
        save_json(output_file, eval_path, agg_results)
        logger.info(f"Saved evaluation to {output_file}")

        return agg_results

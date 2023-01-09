import csv
import os
import pandas as pd
from datetime import datetime
from .utils import logger
from ..validation_data import QEXPDomainData
from gamechangerml.src.search.query_expansion.qe import QE
from gamechangerml.src.search.query_expansion.utils import remove_original_kw
from gamechangerml.src.utilities.test_utils import timestamp_filename, get_user, clean_nans
from gamechangerml.src.utilities import save_json


class QexpEvaluator:
    def __init__(
        self,
        qe_model_dir,
        qe_files_dir,
        method,
        topn,
        threshold,
        min_tokens,
        model=None,
    ):

        self.model_path = qe_model_dir
        if model:
            self.QE = model
        else:
            self.QE = QE(qe_model_dir, qe_files_dir, method)

        self.data = QEXPDomainData().data
        self.topn = topn
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.results = self.eval()

    def predict(self):

        columns = ["query", "expected", "received", "any_match"]
        csv_filename = os.path.join(
            self.model_path, timestamp_filename("qe_domain", ".csv")
        )
        with open(csv_filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(columns)

            query_count = 0
            num_matching = 0
            num_expected = 0
            num_results = 0
            for query, expected in self.data.items():
                logger.info("Query {}: {}".format(str(query_count), query))
                results = self.QE.expand(
                    query, self.topn, self.threshold, self.min_tokens
                )
                results = remove_original_kw(results, query)
                num_results += len(results)
                num_matching += len(set(expected).intersection(results))
                num_expected += min(len(results), self.topn)
                any_match = bool(num_matching)
                row = [[str(query), str(expected), str(results), str(any_match)]]
                csvwriter.writerows(row)
                query_count += 1

        precision = num_matching / num_results
        recall = num_matching / num_expected

        return pd.read_csv(csv_filename), precision, recall

    def eval(self):

        df, precision, recall = self.predict()

        # get overall stats
        num_queries = df.shape[0]

        user = get_user(logger)

        agg_results = {
            "user": user,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_path.split("/")[-1],
            "validation_data": "QE_domain",
            "query_count": clean_nans(num_queries),
            "precision": clean_nans(precision),
            "recall": clean_nans(recall),
        }

        output_file = timestamp_filename("qe_model_eval", ".json")
        save_json(output_file, self.model_path, agg_results)
        logger.info(
            f"Saved evaluation to {str(os.path.join(self.model_path, output_file))}"
        )

        return agg_results

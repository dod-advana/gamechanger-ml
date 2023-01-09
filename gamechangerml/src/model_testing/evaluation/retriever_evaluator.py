import os
import csv
import pandas as pd
from datetime import datetime
from .transformer_evaluator import TransformerEvaluator
from .utils import LOCAL_TRANSFORMERS_DIR, logger
from ..metrics import (
    reciprocal_rank,
    average_precision,
    get_MRR,
    get_MAP,
    get_recall,
    get_optimum_threshold,
)
from gamechangerml.src.utilities import save_json
from gamechangerml.src.utilities.test_utils import timestamp_filename, get_user

retriever_k = 5


class RetrieverEvaluator(TransformerEvaluator):
    def __init__(
        self,
        encoder_model_name,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        use_gpu=False,
    ):

        super().__init__(transformer_path, use_gpu)

        self.encoder_model_name = encoder_model_name
        self.model_path = os.path.join(encoder_model_name, transformer_path)

    def make_index(self, encoder, corpus_path, index_path, files_to_use=None):

        return encoder.index_documents(corpus_path, index_path, files_to_use)

    def predict(self, data, index, retriever, eval_path, k):

        columns = [
            "index",
            "queries",
            "top_expected_ids",
            f"results@{k}",
            "hits",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "reciprocal_rank",
            "average_precision",
        ]
        ## make name for the csv of results
        if "/" in index:
            fname = index.split("/")[-1]
        else:
            fname = index
        csv_filename = os.path.join(
            eval_path, timestamp_filename(fname, ".csv")
        )
        logger.info(f"Making a csv of test results, saved at: {csv_filename}")

        # make the csv
        with open(csv_filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(columns)

            # collect metrics for each query made + results generated
            hit_scores = []
            no_hit_scores = []
            query_count = tp = tn = fp = fn = total_expected = 0
            for idx, query in data.queries.items():
                logger.info("\n\nQ-{}: {}".format(query_count, query))
                doc_results = retriever.retrieve_topn(
                    query, num_results=k
                )  # returns results ordered highest - lowest score
                doc_texts = [x["text"] for x in doc_results]
                doc_ids = [x["id"] for x in doc_results]
                doc_scores = [x["score"] for x in doc_results]
                if fname != "msmarco_index":
                    doc_ids = [".".join(i.split(".")[:-1]) for i in doc_ids]
                logger.info(
                    f"retrieved: {str(doc_texts)}, {str(doc_ids)}, {str(doc_scores)}"
                )
                expected_ids = data.relations[
                    idx
                ]  # collect the expected results (ground truth)
                if type(expected_ids) == str:
                    expected_ids = [expected_ids]
                expected_docs = [data.collection[x] for x in expected_ids]
                expected_docs = list(
                    set([i.split(".pdf")[0] for i in expected_docs])
                )
                logger.info(f"expected: {str(expected_docs)}")
                total_expected += min(
                    len(expected_docs), k
                )  # if we have more than k expected, set this to k
                # collect ordered metrics
                recip_rank = reciprocal_rank(doc_ids, expected_docs)
                avg_p = average_precision(doc_ids, expected_docs)

                # collect non-ordered metrics
                hits = []
                true_pos = false_pos = 0  # no negative samples to test against
                for eid in set(doc_ids):
                    hit = {}
                    if eid in expected_docs:  # we have a hit
                        rank = doc_ids.index(eid)
                        hit["rank"] = rank
                        hit["match"] = eid
                        hit["score"] = doc_scores[rank]
                        hits.append(hit)
                        true_pos += 1
                    else:
                        false_pos += 1
                if (
                    len(doc_ids) < k
                ):  # if there are not k predictions, there are pred negatives
                    remainder = k - len(doc_ids)
                    false_neg = min(
                        len(
                            [i for i in expected_docs if i not in doc_ids],
                            remainder,
                        )
                    )
                    true_neg = min(
                        (k - len(expected_docs)), (k - len(doc_ids))
                    )
                else:  # if there are k predictions, there are no predicted negatives
                    false_neg = true_neg = 0
                if len(hits) > 0:
                    hit_scores.append(hits[0]["score"])
                else:
                    no_hit_scores.append(doc_scores[0])
                fn += false_neg
                tn += true_neg
                tp += true_pos
                fp += false_pos
                logger.info(
                    f"Metrics: fn: {str(fn)}, fp: {str(fp)}, tn: {str(tn)}, tp: {str(tp)}"
                )
                # save metrics to csv
                row = [
                    [
                        str(query_count),
                        str(query),
                        str(expected_docs),
                        str(doc_results),
                        str(hits),
                        str(true_pos),
                        str(false_pos),
                        str(false_neg),
                        str(true_neg),
                        str(recip_rank),  # reciprocal rank
                        str(avg_p),  # average precision
                    ]
                ]
                csvwriter.writerows(row)
                query_count += 1

        return (
            pd.read_csv(csv_filename),
            tp,
            tn,
            fp,
            fn,
            total_expected,
            hit_scores,
            no_hit_scores,
        )

    def eval(
        self,
        data,
        index,
        retriever,
        data_name,
        eval_path,
        model_name,
        k=retriever_k,
    ):

        (
            df,
            tp,
            tn,
            fp,
            fn,
            total_expected,
            hit_scores,
            no_hit_scores,
        ) = self.predict(data, index, retriever, eval_path, k)
        num_queries = df["queries"].shape[0]
        if num_queries > 0:
            _mrr = get_MRR(list(df["reciprocal_rank"].map(float)))
            _map = get_MAP(list(df["average_precision"].map(float)))
            recall = get_recall(
                true_positives=tp, false_negatives=(total_expected - tp)
            )
            best_threshold, max_score = get_optimum_threshold(
                hit_scores, no_hit_scores
            )
        else:
            _mrr = _map = recall = 0

        user = get_user(logger)

        agg_results = {
            "user": user,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "index": index,
            "validation_data": data_name,
            "query_count": num_queries,
            "k": k,
            "MRR": _mrr,
            "mAP": _map,
            "recall": recall,
            "best_f1": max_score,
            "best_threshold": best_threshold,
        }

        logger.info(f"** Eval Results: {str(agg_results)}")
        output_file = timestamp_filename("retriever_eval", ".json")
        save_json(output_file, eval_path, agg_results)
        logger.info(
            f"Saved evaluation to {str(os.path.join(eval_path, output_file))}"
        )

        return agg_results

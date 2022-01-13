import os
import numpy as np
import pandas as pd
import csv
import math
from datetime import datetime
from gamechangerml.src.search.sent_transformer.model import (
    SentenceEncoder,
    SentenceSearcher,
    SimilarityRanker,
)
from gamechangerml.src.search.QA.QAReader import DocumentReader as QAReader
from gamechangerml.src.search.query_expansion.qe import QE
from gamechangerml.src.search.query_expansion.utils import remove_original_kw
from gamechangerml.configs.config import (
    QAConfig,
    EmbedderConfig,
    SimilarityConfig,
    QexpConfig,
    ValidationConfig,
)
from gamechangerml.src.utilities.text_utils import normalize_answer
from gamechangerml.src.utilities.test_utils import *
from gamechangerml.src.model_testing.validation_data import (
    SQuADData,
    NLIData,
    MSMarcoData,
    QADomainData,
    RetrieverGSData,
    UpdatedGCRetrieverData,
    QEXPDomainData,
)
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.src.model_testing.metrics import *
from gamechangerml.api.utils.logger import logger
import signal
import torch

retriever_k = 5

init_timer()
model_path_dict = get_model_paths()
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
SENT_INDEX_PATH = model_path_dict["sentence"]


class TransformerEvaluator:
    def __init__(self, transformer_path=LOCAL_TRANSFORMERS_DIR, use_gpu=False):

        self.transformer_path = transformer_path
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = use_gpu
        else:
            self.use_gpu = False


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
                transformer_path, model_name, qa_type, nbest, null_threshold, use_gpu
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
                        "Q-{}: {}".format(query_count, query["question"]))
                    actual = query["expected"]
                    context = query["search_context"]
                    if type(context) == str:
                        context = [context]
                    prediction = self.model.answer(
                        query["question"], context)[0]
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
        self.eval_path = check_directory(
            os.path.join(self.model_path, "evals_squad"))
        self.results = self.eval(data=self.data, eval_path=self.eval_path)


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
        self.eval_path = check_directory(
            os.path.join(self.model_path, "evals_gc"))
        self.results = self.eval(data=self.data, eval_path=self.eval_path)


class RetrieverEvaluator(TransformerEvaluator):
    def __init__(
        self, encoder_model_name, transformer_path=LOCAL_TRANSFORMERS_DIR, use_gpu=False
    ):

        super().__init__(transformer_path, use_gpu)

        self.encoder_model_name = encoder_model_name
        self.model_path = os.path.join(encoder_model_name, transformer_path)

    def make_index(self, encoder, corpus_path):

        return encoder.index_documents(corpus_path)

    def predict(self, data, index, retriever, eval_path, k):

        columns = [
            "index",
            "queries",
            "top_expected_ids",
            "hits",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "reciprocal_rank",
            "average_precision",
            "precision@{}".format(k),
            "recall@{}".format(k),
        ]
        fname = index.split("/")[-1]
        csv_filename = os.path.join(
            eval_path, timestamp_filename(fname, ".csv"))
        with open(csv_filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(columns)

            # collect metrics for each query made + results generated
            query_count = tp = tn = fp = fn = total_expected = 0
            for idx, query in data.queries.items():
                logger.info("\n\nQ-{}: {}".format(query_count, query))
                doc_results = retriever.retrieve_topn(
                    query, num_results=k
                )  # returns results ordered highest - lowest score
                doc_texts = [x["text"] for x in doc_results]
                doc_ids = [x["id"] for x in doc_results]
                doc_scores = [x["score"] for x in doc_results]
                if index != "msmarco_index":
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
                    set([i.split(".pdf")[0] for i in expected_docs]))
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
                if (
                    len(doc_ids) < k
                ):  # if there are not k predictions, there are pred negatives
                    remainder = k - len(doc_ids)
                    false_neg = min(
                        len([i for i in expected_docs if i not in doc_ids], remainder)
                    )
                    true_neg = min((k - len(expected_docs)),
                                   (k - len(doc_ids)))
                else:  # if there are k predictions, there are no predicted negatives
                    false_neg = true_neg = 0
                fn += false_neg
                tn += true_neg
                tp += true_pos
                logger.info(
                    f"Metrics: fn: {str(fn)}, tn: {str(tn)}, tp: {str(tp)}")
                # save metrics to csv
                row = [
                    [
                        str(query_count),
                        str(query),
                        str(expected_docs),
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

        return pd.read_csv(csv_filename), tp, tn, fp, fn, total_expected

    def eval(
        self, data, index, retriever, data_name, eval_path, model_name, k=retriever_k
    ):

        df, tp, tn, fp, fn, total_expected = self.predict(
            data, index, retriever, eval_path, k
        )
        num_queries = df["queries"].shape[0]
        if num_queries > 0:
            _mrr = get_MRR(list(df["reciprocal_rank"].map(float)))
            _map = get_MAP(list(df["average_precision"].map(float)))
            recall = get_recall(
                true_positives=tp, false_negatives=(total_expected - tp)
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
        }

        logger.info(f"** Eval Results: {str(agg_results)}")
        output_file = timestamp_filename("retriever_eval", ".json")
        save_json(output_file, eval_path, agg_results)
        logger.info(
            f"Saved evaluation to {str(os.path.join(eval_path, output_file))}")

        return agg_results


class MSMarcoRetrieverEvaluator(RetrieverEvaluator):
    def __init__(
        self,
        encoder_model_name,
        sim_model_name,
        min_token_len,
        return_id,
        verbose,
        encoder=None,
        retriever=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        index="sent_index_MSMARCO",
        use_gpu=False,
        data_name="msmarco",
    ):

        super().__init__(transformer_path, encoder_model_name, use_gpu)
        logger.info("Model path: {}".format(self.model_path))
        self.index_path = os.path.join(
            os.path.dirname(transformer_path), index)
        if not os.path.exists(self.index_path):
            logger.info("MSMARCO index path doesn't exist.")
            logger.info(
                "Making new embeddings index at {}".format(
                    str(self.index_path))
            )
            os.makedirs(self.index_path)
            if encoder:
                self.encoder = encoder
            else:
                self.encoder = SentenceEncoder(
                    encoder_model_name=encoder_model_name,
                    min_token_len=min_token_len,
                    return_id=return_id,
                    verbose=verbose,
                    use_gpu=use_gpu,
                )
            self.make_index(
                encoder=self.encoder, corpus_path=None, index_path=self.index_path
            )
        self.data = MSMarcoData()
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = SentenceSearcher(
                sim_model_name=sim_model_name,
                index_path=self.index_path,
                transformer_path=transformer_path,
            )
        self.eval_path = check_directory(
            os.path.join(self.index_path, "evals_msmarco"))
        logger.info("Evals path: {}".format(self.eval_path))
        self.results = self.eval(
            data=self.data,
            index=index,
            retriever=self.retriever,
            data_name=data_name,
            eval_path=self.eval_path,
            model_name=encoder_model_name,
        )


class IndomainRetrieverEvaluator(RetrieverEvaluator):
    def __init__(
        self,
        encoder_model_name,
        sim_model_name,
        min_token_len,
        return_id,
        verbose,
        data_level,
        create_index=True,
        data_path=None,
        encoder=None,
        retriever=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        index=SENT_INDEX_PATH,
        use_gpu=False,
    ):

        super().__init__(transformer_path, encoder_model_name, use_gpu)

        self.model_path = os.path.join(transformer_path, encoder_model_name)
        if not index:
            logger.info("No index provided for evaluating.")
            if create_index:
                self.index_path = os.path.join(
                    os.path.dirname(transformer_path), "sent_index_TEST"
                )
                logger.info(
                    "Making new embeddings index at {}".format(
                        str(self.index_path))
                )
                if not os.path.exists(self.index_path):
                    os.makedirs(self.index_path)
                if encoder:
                    self.encoder = encoder
                else:
                    self.encoder = SentenceEncoder(
                        encoder_model_name=encoder_model_name,
                        min_token_len=min_token_len,
                        return_id=return_id,
                        verbose=verbose,
                        use_gpu=use_gpu,
                    )
                self.make_index(
                    encoder=self.encoder,
                    corpus_path=ValidationConfig.DATA_ARGS["test_corpus_dir"],
                    index_path=self.index_path,
                )
        else:
            self.index_path = os.path.join(
                os.path.dirname(transformer_path), index)

        if self.index_path:
            self.doc_ids = open_txt(os.path.join(
                self.index_path, "doc_ids.txt"))
            if retriever:
                self.retriever = retriever
            else:
                self.retriever = SentenceSearcher(
                    sim_model_name=sim_model_name,
                    index_path=self.index_path,
                    transformer_path=transformer_path,
                )
            self.eval_path = check_directory(
                os.path.join(self.index_path, "evals_gc", data_level)
            )
            self.data = UpdatedGCRetrieverData(
                available_ids=self.doc_ids, level=data_level, data_path=data_path
            )
            self.results = self.eval(
                data=self.data,
                index=index,
                retriever=self.retriever,
                data_name=data_level,
                eval_path=self.eval_path,
                model_name=encoder_model_name,
            )


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
        csv_filename = os.path.join(
            eval_path, timestamp_filename("nli_eval", ".csv"))
        df.to_csv(csv_filename)

        # get overall stats
        all_accuracy = np.round(df["match"].mean(), 2)
        top_accuracy = np.round(
            df[df["expected_rank"] == 0]["match"].mean(), 2)

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
        logger.info(
            f"Saved evaluation to {str(os.path.join(eval_path, output_file))}")

        return agg_results


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
        self.eval_path = check_directory(
            os.path.join(self.model_path, "evals_nli"))
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
        df["match"] = np.where(df["predicted_rank"] ==
                               df["expected_rank"], 1, 0)

        return df


class GCSimEvaluator(SimilarityEvaluator):
    def __init__(
        self,
        sim_model_name,
        model=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        use_gpu=False,
    ):
        # TODO: add in-domain GC dataset for testing sim model (using pos/neg samples/ranking from search)

        super().__init__(sim_model_name, model, transformer_path, use_gpu)

        # self.data = NLIData(sample_limit)
        self.eval_path = check_directory(
            os.path.join(self.model_path, "evals_gc"))
        # self.results = self.eval(predictions=self.predict_nli(), eval_path=self.eval_path)


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
                row = [[str(query), str(expected),
                        str(results), str(any_match)]]
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

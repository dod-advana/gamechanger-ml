from gamechangerml.src.search.sent_transformer import SentenceSearcher


class DocCompareSentenceSearcher(SentenceSearcher):

    # Metadata for the model these scores were derived from
    # {"user": null, "date_started": "2022-04-29 16:06:06", "date_finished": "2022-04-29 19:52:52", "doc_id_count": 1495122, "corpus_name": "/opt/app-root/src/gamechangerml/corpus", "encoder_model": "multi-qa-MiniLM-L6-cos-v1"}
    DEFAULT_SCORES = [
        [0.8, "High"],
        [0.5, "Medium"],
        [0.4, "Low"],
        [0.0, "Very Low"],
    ]
    DEFAULT_CUTOFF = 0.25

    def __init__(self, index_path, sim_model_path):
        super().__init__(index_path, sim_model_path)

    def get_score_display(self, score):
        """Get the display to show for the given score.

        Args:
            score (float): Float in (0,1)

        Returns:
            str
        """
        for threshold, display in self.DEFAULT_SCORES:
            if score > threshold:
                return display

    def retrieve_topn(
        self, query, n=10, use_sim_ranker=True, threshold="auto"
    ):
        top_n = super().retrieve_topn(query, n, use_sim_ranker, threshold)
        for doc in top_n:
            doc["score_display"] = self.get_score_display(doc["score"])

        return top_n

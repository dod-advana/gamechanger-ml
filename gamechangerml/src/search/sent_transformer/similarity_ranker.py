from txtai.pipeline import Similarity


class SimilarityRanker:
    """The Similarity Ranker model is used to compute similarity between a
    query and a list of text.

    Args:
        model_path (str): Path to the similarity model to load.

    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = Similarity(model_path)

    def rank(self, query, texts):
        """Compute the similarity between query and list of text.

        Returns a list of (id, score) sorted by highest score, where id is the
        index in texts.

        This method supports query as a string or a list. If the input is a
        string, the return type is a 1D list of (id, score). If text is a list,
        a 2D list of (id, score) is returned with a row per string.

        Args:
            query (str or list)
            texts (list of str): Texts to rank

        Returns:
            list of (id, score)
        """
        return self.model(query, texts)

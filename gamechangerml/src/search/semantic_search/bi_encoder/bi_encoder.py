from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Union, Callable
from os import path
from gamechangerml.src.utilities import configure_logger
from .util import BiEncoderPreprocessor


class BiEncoder:
    """Bi-encoder retrieval model for semantic search.

    Args:
        model_name_or_path (str): If it is a filepath on disc, it loads the
            model from that path. If it is not a path, it first tries to
            download a pre-trained SentenceTransformer model. If that fails,
            tries to construct a model from Huggingface models repository with
            that name.
        similarity_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            Function for computing scores between text embeddings.
        normalize_embeddings (bool): Whether or not to normalize the text
            embeddings. Use True if the similarity function is dot-product.
        use_gpu (bool): Whether to use GPU.
        logger (logging.Logger or None, optional): If None, creates a logger
            using configure_logger from gamechangerml.src.utilities.
    """

    def __init__(
        self,
        model_name_or_path: str,
        similarity_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        normalize_embeddings: bool,
        use_gpu: bool,
        logger=None,
    ):
        self._embedder = SentenceTransformer(model_name_or_path)
        self._similarity_func = similarity_func

        # Normalizing the embeddings allows us to use util.dot_score which is
        # faster than util.cos_sim.
        self._normalize_embeddings = normalize_embeddings

        self._corpus_embeddings = None
        self._use_gpu = use_gpu
        self._logger = logger if logger else configure_logger()

    def load_corpus_embeddings(self, filepath: str, overwrite: bool = True):
        """Load corpus embeddings from a file and store them as an attribute
        of the object.

        Args:
            filepath (str): Path to the embeddings file.
            overwrite (bool, optional): True to overwrite the object's existing
                corpus embeddings, False otherwise. Defaults to True.
        """
        if self._corpus_embeddings is not None and not overwrite:
            return

        self._corpus_embeddings = torch.load(filepath)

        if self.use_gpu():
            self._corpus_embeddings = self._corpus_embeddings.to("cuda")

    def encode(
        self,
        texts: List[str],
        save_path: Union[str, None] = None,
        overwrite: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """Compute text embeddings.

        Args:
            texts (List[str]): The texts to embed.
            save_path (Union[str, None], optional): If not None, the path to
                save the embeddings to. Defaults to None.
            overwrite (bool, optional): Only applicable when a save_path is
                given. If True, will overwrite any existing embeddings at the
                path. Otherwise, will concatenate existing embeddings with the
                new embeddings. Defaults to False.

        Returns:
            Union[List[torch.Tensor], torch.Tensor]
        """
        texts = [BiEncoderPreprocessor.process(text) for text in texts]

        embeddings = self._embedder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self._normalize_embeddings,
        )

        if save_path is not None:
            self.save_embeddings(embeddings, save_path, overwrite)

        return embeddings

    def save_embeddings(
        self, embeddings: torch.Tensor, save_path: str, overwrite: bool
    ):
        """Save embeddings to a file.

        Args:
            embeddings (torch.Tensor): The embeddings to save.
            save_path (str): Path to save the embeddings to.
            overwrite (bool, optional): If True, will overwrite any existing
                embeddings at the path. Otherwise, will concatenate
                existing embeddings with the new embeddings.
        """
        if not overwrite and path.isfile(save_path):
            previous_embeddings = torch.load(save_path)
            embeddings = torch.cat([embeddings, previous_embeddings])

        torch.save(embeddings, save_path)

    def semantic_search(self, query: str, top_k: int) -> List[dict]:
        """Use the bi-encoder model to retrieve the query's most relevant
        documents in the corpus.

        Similarity between the query and corpus entries is computed using the
        similarity function defined during initialization.

        Args:
            query (str): The query to find relevant passages for.
            top_k (int): Retrieve the top k matching entries.

        Raises:
            Exception: If the corpus embeddings were not loaded yet. See
                load_corpus_embeddings().

        Returns:
            List[dict]: A sorted list with decreasing similarity scores. Entries
                are dictionaries with the keys 'corpus_id' and 'score'
        """
        if self._corpus_embeddings is None:
            raise Exception(
                "You must load corpus embeddings (load_corpus_embeddings()) before performing semantic search."
            )
        query = BiEncoderPreprocessor.process(query, True)
        query_embedding = self.encode([query])

        if self._normalize_embeddings:
            query_embedding = util.normalize_embeddings(query_embedding)

        if self.use_gpu():
            query_embedding = query_embedding.to("cuda")

        hits = util.semantic_search(
            query_embedding,
            self._corpus_embeddings,
            score_function=self._similarity_func,
            top_k=top_k,
        )

        if len(hits) > 0:
            return hits[0]
        else:
            return []

    def use_gpu(self):
        if self._use_gpu:
            if torch.cuda.is_available():
                return True
            else:
                self._logger.warning("Cannot use GPU. CUDA not available.")

        return False

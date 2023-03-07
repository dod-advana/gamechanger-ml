"""Class definition for SemanticSearch, which is used in the ML API
/transSentenceSearch endpoint. Also serves as a base class for DocumentComparison."""

from txtai.embeddings import Embeddings
from txtai.ann import ANN
from threading import current_thread
from numpy import empty, float32, save as np_save, interp
from pandas import DataFrame, read_csv
from pickle import load as load_pickle
from os import remove as delete_file
from os.path import join
from typing import List, Tuple, Union, Literal

from gamechangerml.api.utils import processmanager
from gamechangerml.configs import SemanticSearchConfig
from gamechangerml.src.utilities import get_most_recent_eval, open_json
from gamechangerml.src.text_handling.process import preprocess
from gamechangerml.src.model_testing.validation_data import MSMarcoData
from .utils import LocalCorpusTokenizer

# TODO: look into re-ranking: https://www.sbert.net/examples/applications/retrieve_rerank/README.html


class SemanticSearch:
    """The idea behind semantic search is to embed all entries in your corpus
    (in our case, paragraphs) into a vector space. (See create_embeddings_index()).

    At search time, the query is embedded into the same vector space and the
    closest embeddings from your corpus are found. These entries should have a
    high semantic overlap with the query. (See search().)

    Searching a large corpus with millions of embeddings can be time-consuming
    if exact nearest neighbor search is used. In that case, Approximate Nearest
    Neighbor (ANN) can be helpful. Here, the data is partitioned into smaller
    fractions of similar embeddings. This index can be searched efficiently and
    the embeddings with the highest similarity (the nearest neighbors) can be
    retrieved within milliseconds, even if you have millions of vectors.

    Args:
        model_path (str): Path to a vectors model supported by
            txtai.embeddings.Embeddings.
        index_directory_path (str): Path to directory for index files.
        load_index_from_file (bool): True to load the index from files in
            `index_directory_path`. If False, you must create the index with
            `create_embeddings_index()` before calling `search()`.
        logger (logging.Logger)
        use_gpu (bool)

    Attributes:
        index_path (str): Path to directory containing index files.
        embeddings_path (str): Path to embeddings.npy file that is created
            during `create_embeddings_index()`.
        ids_path (str): Path to doc_ids.txt file that is created during
            `create_embeddings_index()`.
        paragraphs_and_ids_path (str): Path to the data.csv file that is created
            during `create_embeddings_index()`.
        paragraphs_and_ids_df (pandas.DataFrame): DataFrame with columns "text"
            and "paragraph_id". Used to look up texts of IDs returned by an
            embeddings search.
        use_gpu (bool): True to use GPU, False otherwise.
        score_display_map (dict or None): If not None, adds "score_display"
            field to results of search() based on this map. Defaults to None.
    """

    def __init__(
        self,
        model_path,
        index_directory_path,
        load_index_from_file,
        logger,
        use_gpu,
    ):
        self.embedder = Embeddings(
            {"method": "transformers", "path": model_path, "use_gpu": use_gpu}
        )
        self.logger = logger

        # This DataFrame can either be created with create_embeddings_index() or
        # loaded from a file with _load_paragraphs_and_ids_df().
        self.paragraphs_and_ids_df = None

        self.score_display_map = self.get_score_display_map()

        self.index_path = index_directory_path
        self.embeddings_path = join(self.index_path, "embeddings.npy")
        self.paragraphs_and_ids_path = join(self.index_path, "data.csv")
        self.ids_path = join(self.index_path, "doc_ids.txt")

        # Populates when "auto" is passed as the `threshold` argument in search().
        self._auto_threshold = None

        if load_index_from_file:
            self.embedder.load(self.index_path)

    def get_score_display_map(self) -> Union[dict, None]:
        # Different implementation in DocumentComparison
        return None

    def get_default_threshold(self) -> float:
        # Different implementation in DocumentComparison
        return SemanticSearchConfig.DEFAULT_THRESHOLD_FLOAT

    def should_include_results_below_threshold(self) -> bool:
        # Different implementation in DocumentComparison
        return SemanticSearchConfig.INCLUDE_RESULTS_BELOW_THRESHOLD

    def get_eval_threshold_multiplier(self) -> float:
        return SemanticSearchConfig.EVAL_THRESHOLD_MULTIPLIER

    def prepare_corpus_for_embedding(self, directory_path, files_to_use=None):
        """Load and prepare the corpus before creating embeddings.

        Args:
            directory_path (str or None): Path to directory containing JSON
                corpus files. If None, creates a test corpus with MSMarcoData.
            files_to_use (list of str or None, optional): List of JSON file
                names to process. If empty, will process all JSON files in the
                directory. Defaults to None.
        """
        if directory_path is None:
            self.logger.info("Creating test corpus with MSMarcoData.")
            return MSMarcoData().corpus
        else:
            corpus = LocalCorpusTokenizer(directory_path, files_to_use)
            return [(id_, " ".join(tokens), None) for tokens, id_ in corpus]

    def create_embeddings_index(
        self,
        corpus: List[Tuple[str, str, any]],
        save_vectors: bool = False,
    ) -> None:
        """Transform the corpus into embeddings vectors and create an
        Approximate Nearest Neighbors (ANN) index.

        Creates the files in the directory at `index_directory_path`:
            - data.csv (contains columns "text" and "paragraph_id")
            - embeddings.npy (only if save_vectors arg is True)
            - doc_ids.txt

        Populates the following object attributes:
            - `embedder` with the embeddings data
            - `paragraphs_and_ids_df` with the data saved to data.csv

        Args:
            corpus (List[List[Tuple): Paragraph IDs and text. Use
                `prepare_corpus_for_embedding()`.
            save_vectors (bool, optional): True to save the embeddings, False
                otherwise. Defaults to False.
        """
        self._update_process_manager_training_status(False)

        self.logger.info("Transforming documents to embeddings vectors.")
        ids, dimensions, stream = self.embedder.model.index(corpus)

        self.logger.info("Loading embeddings into memory.")
        embeddings = empty((len(ids), dimensions), dtype=float32)
        with open(stream, "rb") as queue:
            for i in range(embeddings.shape[0]):
                embeddings[i] = load_pickle(queue)

        self.logger.info("Removing temporary stream file.")
        delete_file(stream)

        if save_vectors:
            self.logger.info("Saving vectors.")
            np_save(self.embeddings_path, embeddings)

        self.logger.info("Saving file with IDs.")
        with open(self.ids_path, "w") as f:
            f.writelines("\n".join(ids))

        self.logger.info("Saving data as csv.")
        df = DataFrame(
            [[x[0], str(x[1])] for x in corpus],
            columns=["text", "paragraph_id"],
        )
        df.to_csv(self.paragraphs_and_ids_path, index=False)
        self.paragraphs_and_ids_df = df

        self.logger.info("Normalizing embeddings.")
        self.embedder.normalize(embeddings)

        self.logger.info("Saving ids & dimensions to embedder config.")
        self.embedder.config["ids"] = ids
        self.embedder.config["dimensions"] = dimensions

        self.logger.info("Creating the ANN model.")
        self.embedder.embeddings = ANN.create(self.embedder.config)

        self.logger.info("Building the embeddings index.")
        self.embedder.embeddings.index(embeddings)

        self.logger.info("Saving the embeddings index.")
        self.embedder.save(self.index_path)

        self._update_process_manager_training_status(True)

    def search(
        self,
        query,
        num_results: int,
        preprocess_query: bool,
        threshold: Union[
            Literal["auto"], float
        ] = SemanticSearchConfig.DEFAULT_THRESHOLD_ARG,
    ):
        """Run a semantic search for the given query.

        The query is embedded into the same vector space as the corpus and the
        closest embeddings from the corpus are found. 

        Args:
            query (str): Search the corpus for text with high semantic overlap
                with this query.
            num_results (int): Number of results to return.
            preprocess_query (bool): True to preprocess the query, False otherwise.
            threshold (float or "auto"): Minimum score for a search result to be
            considered a passing result. If auto, will calculate the threshold
            using the most recent evaluation data.

        Returns:
            list of dict: List of dictionaries of the format:
                {
                    "id": str,
                    "text": str,
                    "text_length": float,
                    "score": float,
                    "passing_result": int (0 or 1),
                    "score_display": str, optional (only included if
                        self.score_display_map is populated, e.g. for
                        DocumentComparison)
                }
        """
        threshold = self._transform_threshold(threshold)

        if preprocess_query:
            query = " ".join(preprocess(query))

        self.logger.info(f"Running semantic search for query: `{query}`.")

        if len(query) <= 2:
            self.logger.info("Query is too short.")
            return []

        ids_and_scores = self.embedder.search(query, limit=num_results)
        texts = [
            self._get_paragraph_by_id(par_id) for par_id, _ in ids_and_scores
        ]

        # Calculate normalized text lengths.
        lengths = [len(text) for text in texts]
        norm_lengths = interp(lengths, (min(lengths), max(lengths)), (0, 0.2))

        # Format results.
        results = [
            {
                "id": ids_and_scores[i][0],
                "text": texts[i],
                "text_length": norm_lengths[i],
                "score": ids_and_scores[i][1],
                "passing_result": int(ids_and_scores[i][1] >= threshold),
            }
            for i in range(len(ids_and_scores))
        ]

        if self.score_display_map:
            for res in results:
                res["score_display"] = self._get_score_display(res["score"])

        # Sort the results by score.
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        if not self.should_include_results_below_threshold():
            results = [res for res in results if res["score"] >= threshold]

        return results

    def _calculate_auto_threshold(self) -> float:
        """Calculate the threshold to use when `auto` is passed. Looks for the
        best threshold value from the most recent evaluation file and scales it
        by a multiplier."""

        if self._auto_threshold is not None:
            return self._auto_threshold

        try:
            directory = join(self.index_path, "evals_gc", "silver")
            filename = get_most_recent_eval(directory)
            file = open_json(filename, directory)
            threshold = (
                float(file["best_threshold"])
                * self.get_eval_threshold_multiplier()
            )
        except Exception:
            threshold = self.get_default_threshold()
            self.logger.exception(
                "Failed to determine auto threshold based on eval file."
            )

        self._auto_threshold = threshold

        return threshold

    def _transform_threshold(self, value) -> float:
        """Transform the threshold arg passed to the init into what will be
        used in search()."""
        if value == "auto":
            return self._calculate_auto_threshold()

        try:
            threshold = float(value)
        except:
            threshold = self.get_default_threshold()
            self.logger.exception(
                f"Failed to convert threshold `{threshold}` to float. Using default."
            )

        if threshold <= 0 or threshold >= 1:
            self.logger.error("Threshold must be in (0,1). Using default.")
            threshold = self.get_default_threshold()

        return threshold

    def _get_score_display(self, score) -> Union[str, None]:
        """Get the `score_display` value (e.g., "High" or "Low") for the given
        score. The value is based on the object's `score_display_map`."""
        for threshold, display in self.score_display_map.items():
            if score > threshold:
                return display

        return None

    def _load_paragraphs_and_ids_df(self) -> None:
        """If the object's `paragraphs_and_ids_df` attribute is empty, it will
        be populated with data from the `data.csv` file which contains
        preprocessed paragraph texts and IDs.
        """
        if self.paragraphs_and_ids_df is not None:
            return

        try:
            self.paragraphs_and_ids_df = read_csv(
                self.paragraphs_and_ids_path, dtype={"paragraph_id": str}
            )
        except Exception:
            self.logger.exception(
                "Failed to load corpus paragraphs and ids. Did you forget to create the index?"
            )

    def _get_paragraph_by_id(self, paragraph_id) -> str:
        """Returns the text of the paragraph with the given paragraph id.

        Args:
            paragraph_id (str): Paragraph ID assigned by the gamechanger-data
                parser pipeline. Ex: `DoD 1325.7-M CH 3.pdf_94`

        Returns:
            str
        """
        self._load_paragraphs_and_ids_df()

        return self.paragraphs_and_ids_df[
            self.paragraphs_and_ids_df["paragraph_id"] == str(paragraph_id)
        ].iloc[0]["text"]

    def _update_process_manager_training_status(self, finished: bool) -> None:
        """Pass True if the training process is finished or False if it is starting."""
        processmanager.update_status(
            processmanager.training,
            int(finished),
            1,
            f"{'Finished' if finished else 'Started'} building embeddings index.",
            thread_id=current_thread().ident,
        )

import os
from gamechangerml.configs import SemanticSearchConfig
from gamechangerml.src.search.semantic_search import SemanticSearch
from gamechangerml.src.utilities import create_directory_if_not_exists
from .retriever_evaluator import RetrieverEvaluator
from .utils import LOCAL_TRANSFORMERS_DIR, logger
from ..validation_data import MSMarcoData


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
            os.path.dirname(transformer_path), index
        )
        if not os.path.exists(self.index_path):
            logger.info("MSMARCO index path doesn't exist.")
            logger.info(
                "Making new embeddings index at {}".format(
                    str(self.index_path)
                )
            )
            os.makedirs(self.index_path)
            if encoder:
                self.encoder = encoder
            else:
                self.encoder = SemanticSearch(
                    self.model_path,
                    self.index_path,
                    False,
                    logger,
                    use_gpu,
                    SemanticSearchConfig.DEFAULT_THRESHOLD_ARG,
                )
            self.make_index(
                encoder=self.encoder,
                corpus_path=None,
            )
        self.data = MSMarcoData()
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = SemanticSearch(
                self.model_path,
                self.index_path,
                True,
                logger,
                use_gpu,
                SemanticSearchConfig.DEFAULT_THRESHOLD_ARG,
            )
        self.eval_path = create_directory_if_not_exists(
            os.path.join(self.index_path, "evals_msmarco")
        )
        logger.info("Evals path: {}".format(self.eval_path))
        self.results = self.eval(
            data=self.data,
            index=index,
            retriever=self.retriever,
            data_name=data_name,
            eval_path=self.eval_path,
            model_name=encoder_model_name,
        )

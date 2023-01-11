from os.path import join
from gamechangerml.src.search.QA.QAReader import DocumentReader as QAReader
from gamechangerml.configs import (
    QAConfig,
    QexpConfig,
    TopicsConfig,
    SemanticSearchConfig,
    DocumentComparisonConfig,
)
from gamechangerml.src.search.semantic_search import SemanticSearch
from gamechangerml.src.search.document_comparison import DocumentComparison
from gamechangerml.src.search.query_expansion import qe
from gamechangerml.src.recommender.recommend import Recommender
from gamechangerml.src.search.embed_reader import sparse
from gamechangerml.api.fastapi.settings import (
    logger,
    TOPICS_MODEL,
    QEXP_JBOOK_MODEL_NAME,
    QEXP_MODEL_NAME,
    WORD_SIM_MODEL,
    LOCAL_TRANSFORMERS_DIR,
    SENT_INDEX_PATH,
    DOC_COMPARE_SENT_INDEX_PATH,
    latest_intel_model_encoder,
    latest_intel_model_trans,
    latest_doc_compare_encoder,
    QA_MODEL,
)
from gamechangerml.src.featurization.word_sim import WordSim
from gamechangerml.src.featurization.topic_modeling import Topics
from gamechangerml.api.utils import processmanager

# A singleton class that loads all of the models.
# All variables and methods are static so you
# reference them by ModelLoader().example_method()


class ModelLoader:
    # private model variables
    def __init__(self):
        __qa_model = None
        __query_expander = None
        __query_expander_jbook = None
        __word_sim = None
        __sparse_reader = None
        __topic_model = None
        __recommender = None
        __document_compare_searcher = None
        __semantic_search = None

    # Get methods for the models. If they don't exist try initializing them.
    def getQA(self):
        if ModelLoader.__qa_model == None:
            logger.warning(
                "qa_model was not set and was attempted to be used. Running init"
            )
            ModelLoader.initQA()
        return ModelLoader.__qa_model

    def getQE(self):
        if ModelLoader.__query_expander == None:
            logger.warning(
                "query_expander was not set and was attempted to be used. Running init"
            )
            ModelLoader.initQE()
        return ModelLoader.__query_expander

    def getQEJbook(self):
        if ModelLoader.__query_expander_jbook == None:
            logger.warning(
                "query_expander was not set and was attempted to be used. Running init"
            )
            ModelLoader.initQEJBook()
        return ModelLoader.__query_expander_jbook

    def getWordSim(self):
        if ModelLoader.__word_sim == None:
            logger.warning(
                "word_sim was not set and was attempted to be used. Running init"
            )
            # ModelLoader.initWordSim()
        return ModelLoader.__word_sim

    def getSemanticSearch(self):
        if ModelLoader.__semantic_search == None:
            logger.warning(
                "semantic_search was not set and was attempted to be used. Running init"
            )
            ModelLoader.initSemanticSearch()
        return ModelLoader.__semantic_search

    def getDocumentCompareSearcher(self):
        if ModelLoader.__document_compare_searcher == None:
            logger.warning(
                "document_compare_searcher was not set and was attempted to be used. Running init"
            )
            ModelLoader.initDocumentCompareSearcher()
        return ModelLoader.__document_compare_searcher

    def getSparse(self):
        return ModelLoader.__sparse_reader

    def getTopicModel(self):
        if ModelLoader.__topic_model is None:
            logger.warning(
                "topic_model was not set and was attempted to be used. Running init"
            )
            ModelLoader.initTopics()
        return ModelLoader.__topic_model

    def getRecommender(self):
        if ModelLoader.__recommender is None:
            logger.warning(
                "recommender was not set and was attempted to be used. Running init"
            )
            ModelLoader.initRecommender()
        return ModelLoader.__recommender

    def set_error(self):
        logger.error("Models cannot be directly set. Must use init methods.")

    # Static variables that use these custom getters defined above.
    # So when ModelLoader().qa_model is referenced getQA is called.
    qa_model = property(getQA, set_error)
    query_expander = property(getQE, set_error)
    query_expander_jbook = property(getQEJbook, set_error)
    sparse_reader = property(getSparse, set_error)
    semantic_search = property(getSemanticSearch, set_error)
    word_sim = property(getWordSim, set_error)
    topic_model = property(getTopicModel, set_error)
    recommender = property(getRecommender, set_error)
    document_compare_searcher = property(getDocumentCompareSearcher, set_error)

    @staticmethod
    def initQA(qa_model_name=QA_MODEL.value):
        """initQA - loads transformer model on start
        Args:
        Returns:
        """
        try:
            logger.info("Starting QA pipeline")
            ModelLoader.__qa_model = QAReader(
                transformer_path=LOCAL_TRANSFORMERS_DIR.value,
                use_gpu=True,
                model_name=qa_model_name,
                **QAConfig.MODEL_ARGS,
            )
            # set cache variable defined in settings.py
            QA_MODEL.value = ModelLoader.__qa_model.READER_PATH
            logger.info("Finished loading QA Reader")
        except OSError:
            logger.error(f"Could not load Question Answer Model")

    @staticmethod
    def initQE(qexp_model_path=QEXP_MODEL_NAME.value):
        """initQE - loads QE model on start
        Args:
        Returns:
        """
        logger.info(f"Loading Pretrained Vector from {qexp_model_path}")
        try:
            ModelLoader.__query_expander = qe.QE(
                qexp_model_path, **QexpConfig.INIT_ARGS
            )
            logger.info("** Loaded Query Expansion Model")
        except Exception as e:
            logger.warning("** Could not load QE model")
            logger.warning(e)

    @staticmethod
    def initQEJBook(qexp_jbook_model_path=QEXP_JBOOK_MODEL_NAME.value):
        """initQE - loads JBOOK QE model on start
        Args:
        Returns:
        """
        logger.info(f"Loading Pretrained Vector from {qexp_jbook_model_path}")
        try:
            ModelLoader.__query_expander_jbook = qe.QE(
                qexp_jbook_model_path, **QexpConfig.INIT_ARGS
            )
            logger.info("** Loaded JBOOK Query Expansion Model")
        except Exception as e:
            logger.warning("** Could not load JBOOK QE model")
            logger.warning(e)

    @staticmethod
    def initWordSim(model_path=WORD_SIM_MODEL.value):
        """initQE - loads QE model on start
        Args:
        Returns:
        """
        logger.info(f"Loading Word Sim Model from {model_path}")
        try:
            ModelLoader.__word_sim = WordSim(model_path)
            logger.info("** Loaded Word Sim Model")
        except Exception as e:
            logger.warning("** Could not load Word Sim model")
            logger.warning(e)

    @staticmethod
    def initSemanticSearch(index_path=SENT_INDEX_PATH.value):
        logger.info("Loading Semantic Search model.")
        try:
            ModelLoader.__semantic_search = SemanticSearch(
                model_path=join(
                    LOCAL_TRANSFORMERS_DIR.value,
                    SemanticSearchConfig.BASE_MODEL,
                ),
                index_directory_path=index_path,
                load_index_from_file=SemanticSearchConfig.LOAD_INDEX_FROM_FILE,
                logger=logger,
                use_gpu=SemanticSearchConfig.USE_GPU,
            )
        except:
            logger.exception("Failed to init Semantic Search model.")
        else:
            latest_intel_model_encoder.value = SemanticSearchConfig.BASE_MODEL
            logger.info(
                f"Finished loading Semantic Search model: {SemanticSearchConfig.BASE_MODEL}."
            )

    @staticmethod
    def initDocumentCompareSearcher(
        index_path=DOC_COMPARE_SENT_INDEX_PATH.value,
    ):
        """Creates a DocumentCompare instance for the /documentCompare endpoint.

        Args:
            index_path (str, optional): Path to the directory of index files.
            Defaults to DOC_COMPARE_SENT_INDEX_PATH.value.
        """
        logger.info(
            f"Loading Document Compare Searcher with index path: {index_path}"
        )
        try:
            ModelLoader.__document_compare_searcher = DocumentComparison(
                model_path=join(
                    LOCAL_TRANSFORMERS_DIR.value,
                    DocumentComparisonConfig.BASE_MODEL,
                ),
                index_directory_path=index_path,
                load_index_from_file=True,
                logger=logger,
                use_gpu=DocumentComparisonConfig.USE_GPU,
            )
            latest_doc_compare_encoder.value = (
                DocumentComparisonConfig.BASE_MODEL
            )
        except Exception as e:
            logger.warning("** Could not load Document Comparison model")
            logger.warning(e)

    @staticmethod
    def initSparse(model_name=latest_intel_model_trans.value):
        try:
            ModelLoader.__sparse_reader = sparse.SparseReader(
                model_name=model_name
            )
            logger.info(f"Sparse Reader: {model_name} loaded")
        except Exception as e:
            logger.warning("** Could not load Sparse Reader")
            logger.warning(e)

    @staticmethod
    def initTopics(model_path=TOPICS_MODEL.value) -> None:
        """initTopics - load topics model on start
        Args:
        Returns:
        """
        try:
            logger.info(f"Loading topic model {model_path}")
            logger.info(TopicsConfig.DATA_ARGS)
            ModelLoader.__topic_model = Topics(directory=model_path)
            logger.info("Finished loading Topic Model")
        except Exception as e:
            logger.warning("** Could not load Topic model")
            logger.warning(e)

    @staticmethod
    def initRecommender():
        """initRecommender - loads recommender class on start
        Args:
        Returns:
        """
        try:
            logger.info("Starting Recommender pipeline")
            ModelLoader.__recommender = Recommender()
            logger.info("Finished loading Recommender")
        except OSError:
            logger.error(f"** Could not load Recommender")

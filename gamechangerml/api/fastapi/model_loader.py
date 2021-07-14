import os
from gamechangerml.src.search.QA.QAReader import DocumentReader as QAReader
from gamechangerml.configs.config import QAConfig
from gamechangerml.src.search.query_expansion import qe
from gamechangerml.src.search.sent_transformer.model import SentenceSearcher
from gamechangerml.src.search.embed_reader import sparse
from gamechangerml.api.fastapi.settings import *

# A singleton class that loads all of the models.
# All variables and methods are static so you 
# reference them by ModelLoader().example_method()
class ModelLoader:
    # private model variables 
    __qa_model = None
    __sentence_trans = None
    __query_expander = None
    __sparse_reader = None

    # Get methods for the models. If they don't exist try initializing them.
    def getQA(self):
        if ModelLoader.__qa_model == None:
            logger.warning("qa_model was not set and was attempted to be used. Running init")
            ModelLoader.initQA()
        return ModelLoader.__qa_model
    
    def getQE(self):
        if ModelLoader.__query_expander == None:
            logger.warning("query_expander was not set and was attempted to be used. Running init")
            ModelLoader.initQE()
        return ModelLoader.__query_expander

    def getSentence_trans(self):
        if ModelLoader.__sentence_trans == None:
            logger.warning("sentence_trans was not set and was attempted to be used. Running init")
            ModelLoader.initSentence()
        return ModelLoader.__sentence_trans

    def getSparse(self):
        return ModelLoader.__sparse_reader

    def set_error(self):
        logger.error("Models cannot be directly set. Must use init methods.")

    # Static variables that use these custom getters defined above.
    # So when ModelLoader().qa_model is referenced getQA is called.
    qa_model = property(getQA,set_error)
    query_expander = property(getQE,set_error)
    sparse_reader = property(getSparse,set_error)
    sentence_trans = property(getSentence_trans,set_error)

    @staticmethod 
    def initQA():
        """initQA - loads transformer model on start
        Args:
        Returns:
        """
        try:
            qa_model_path = os.path.join(
                LOCAL_TRANSFORMERS_DIR.value, "bert-base-cased-squad2")
            logger.info("Starting QA pipeline")
            ModelLoader.__qa_model = QAReader(qa_model_path, use_gpu=True, **QAConfig.MODEL_ARGS)
            # set cache variable defined in settings.py
            latest_qa_model.value = qa_model_path
            logger.info("Finished loading QA Reader")
        except OSError:
            logger.error(f"Could not load Question Answer Model")
    
    @staticmethod 
    def initQE(qexp_model_path=QEXP_MODEL_NAME.value):
        """initQE - loads QE model on start
        Args:
        Returns:
        """
        logger.info(f"Loading Query Expansion Model from {qexp_model_path}")
        try:
            ModelLoader.__query_expander = qe.QE(
                qexp_model_path, method="emb", vocab_file="word-freq-corpus-20201101.txt"
            )
            logger.info("** Loaded Query Expansion Model")
        except Exception as e:
            logger.warning("** Could not load QE model")
            logger.warning(e)

    @staticmethod
    def initSentence(
        index_path=SENT_INDEX_PATH.value, transformer_path=LOCAL_TRANSFORMERS_DIR.value
    ):
        """
        initQE - loads Sentence Transformers on start
        Args:
        Returns:
        """
        # load defaults
        encoder_model = os.path.join(
            transformer_path, "msmarco-distilbert-base-v2")
        logger.info(f"Using {encoder_model} for sentence transformer")
        sim_model = os.path.join(transformer_path, "distilbart-mnli-12-3")
        logger.info(f"Loading Sentence Transformer from {sim_model}")
        logger.info(f"Loading Sentence Index from {index_path}")
        try:
            ModelLoader.__sentence_trans = SentenceSearcher(
                index_path=index_path,
                sim_model=sim_model,
            )
            # set cache variable defined in settings.py
            latest_intel_model_sent.value  = {"encoder": encoder_model, "sim": sim_model}
            logger.info("** Loaded Sentence Transformers")
        except Exception as e:
            logger.warning("** Could not load Sentence Transformer model")
            logger.warning(e)
    
    @staticmethod
    def initSparse(model_name = latest_intel_model_trans.value):
        try:
            ModelLoader.__sparse_reader= sparse.SparseReader(
                model_name=model_name)
            logger.info(f"Sparse Reader: {model_name} loaded")
        except Exception as e:
            logger.warning("** Could not load Sparse Reader")
            logger.warning(e)
            
    # Currently deprecated
    @staticmethod
    def initTrans():
        """initTrans - loads transformer model on start
        Args:
        Returns:
        """
        try:
            model_name = os.path.join(
                LOCAL_TRANSFORMERS_DIR.value, "distilbert-base-uncased-distilled-squad"
            )
            # not loading due to ram and deprecation
            # logger.info(f"Attempting to load in BERT model default: {model_name}")
            logger.info(
                f"SKIPPING LOADING OF TRANSFORMER MODEL FOR INTELLIGENT SEARCH: {model_name}"
            )
            # ModelLoader.__sparse_reader = sparse.SparseReader(model_name=model_name)
            # logger.info(
            #    f" ** Successfully loaded BERT model default: {model_name}")
            logger.info(f" ** Setting Redis model to {model_name}")
            # set cache variable defined in settings.py
            latest_intel_model_trans.value = model_name
        except OSError:
            logger.error(f"Could not load BERT Model {model_name}")
            logger.error(
                "Check if BERT cache is in correct location: tranformer_cache/ above root directory."
            )
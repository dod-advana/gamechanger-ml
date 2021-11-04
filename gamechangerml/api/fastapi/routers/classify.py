import pandas as pd
from fastapi import APIRouter, Response, status
import time

# must import sklearn first or you get an import error
from gamechangerml.src.search.query_expansion.utils import remove_original_kw
from gamechangerml.src.featurization.keywords.extract_keywords import get_keywords
from gamechangerml.src.text_handling.process import preprocess
from gamechangerml.api.fastapi.version import __version__

# from gamechangerml.models.topic_models.tfidf import bigrams, tfidf_model
from gamechangerml.src.featurization.summary import GensimSumm
from gamechangerml.api.fastapi.settings import *
from gamechangerml.api.fastapi.model_loader import ModelLoader

from gamechangerml.configs.config import QexpConfig

router = APIRouter()
MODELS = ModelLoader()


@router.post("/transformerClassify", status_code=200)
async def transformer_infer(corpus: dict, response: Response) -> dict:
    """transformer_infer - endpoint for transformer inference
   Args:
        query: dict; format of query
            {"text": "i am text"}
        Response: Response class; for status codes(apart of fastapi do not need to pass param)

        #TODO update extra arguments (model type?)
        extractType: topics, keywords, or summary

    Returns:
        results: dict; results of inference
    """
    logger.debug("TRANSFORMER - predicting text: " + str(query))
    results = {}
    try:
        pdocs = []
        rdocs = []
        classification_list = []
    #TODO unravel json
        #docs= {json element containing documents}
    # TODO Determine if pdoc or rdoc    (number of columns? or tagging?)
        #if number of keys in doc dict ==
            #pdocs.append(doc)
        #pd.read_json(docs)


        #TODO clean text
        # call clean_text() from utils
        # call encoder()

        # pass data through classifier model

        #for record in corpus:
            #.predict(record)
        # construct return payload


        #results = MODELS.{model_predictor}.predict(text)
        logger.info(results)
    except Exception:
        logger.error(f"Unable to get results from transformer for {query}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results

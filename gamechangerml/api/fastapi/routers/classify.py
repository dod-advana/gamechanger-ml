import json

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
from gamechangerml.src.text_classif.predictor import Predictor


from gamechangerml.configs.config import QexpConfig

router = APIRouter()
MODELS = ModelLoader()


@router.post("/transformerClassify", status_code=200)
async def transformer_infer(payload: dict, response: Response) -> dict:
    """transformer_infer - endpoint for transformer inference
   Args:
        corpus: dict; format of query
            {"text": "i am text"}
        Response: Response class; for status codes(apart of fastapi do not need to pass param)

        extractType: topics, keywords, or summary

    Returns:
        results: dict; results of inference
    """
    logger.debug("TRANSFORMER - predicting text: " + str(query))
    results = {}

    #TODO RAC update to table columns
    text_col_dict = {
        "pdoc": ["Program_Description", "Budget_Justification"],
        "rdoc": ["Project_Mission_Description", "PE_Mission_Description_and_Budget_Justification", "Project_Title",
                 "Program_Element_Title",
                 "Project_Notes", "Project_Aquisition_Strategy", "Project_Perfromance_Metircs",
                 "Other_program_funding_summary_remarks"]
    }

    label_mapping = {
        0: "Not AI",
        1: "AI Enabled",
        2: "Core AI",
        3: "AI Enabling"
    }

    try:
        text_list=[]
    #TODO unravel json
    #TODO Determine if pdoc or rdoc    (number of columns? or tagging?)

        #how does this separate for each record?
        for concat_col in text_col_dict[record['budget_type']]:
            for doc in payload:
                combined_text += doc[concat_col]
            # aggregate text columns
            text_list.append(combined_text)




        #TODO clean text
        def clean_text(text):
            """
            Performs the following transformation on a string that is passed in:
            1. Lowercase the text
            2. Replaces /(){}\[\]\|@,;#+_ characters with spaces
            3. Removes any non numeric or lowercase alphabetical characters
            4. Removes stopwords (from nltk)
            :param text: (str) Text to be cleaned
            :return: (str) Cleaned text
            """
            import re
            from nltk.corpus import stopwords
            REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#+_]')
            BAD_SYMBOLS_RE = re.compile('[^0-9a-z ]')
            STOPWORDS = set(stopwords.words('english'))
            text = text.lower()  # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
            text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
            text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwords from text
            return text

        text_list.map(lambda text: clean_text(text))
        # call encoder()


        # pass data through classifier model
            # results = MODELS.classify_trans.predict(text)   #What does predictor return?


        # construct return payload
            #results.map(lambda num_class: label_mapping[num_class])


        logger.info(results)
    except Exception:
        logger.error(f"Unable to get results from transformer for {query}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results

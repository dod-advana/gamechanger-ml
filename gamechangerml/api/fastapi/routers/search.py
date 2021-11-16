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


@router.post("/transformerSearch", status_code=200)
async def transformer_infer(query: dict, response: Response) -> dict:
    """transformer_infer - endpoint for transformer inference
    Args:
        query: dict; format of query
            {"query": "test", "documents": [{"text": "...", "id": "xxx"}, ...]
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        results: dict; results of inference
    """
    logger.debug("TRANSFORMER - predicting query: " + str(query))
    results = {}
    try:
        results = MODELS.sparse_reader.predict(query)
        logger.info(results)
    except Exception:
        logger.error(f"Unable to get results from transformer for {query}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results


@router.post("/textExtractions", status_code=200)
async def textExtract_infer(query: dict, extractType: str, response: Response) -> dict:
    """textExtract_infer - endpoint for sentence transformer inference
    Args:
        query: dict; format of query
            {"text": "i am text"}
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
        extractType: topics, keywords, or summary
    Returns:
        results: dict; results of inference
    """
    results = {}
    try:
        query_text = query["text"]
        results["extractType"] = extractType
        if extractType == "topics":
            logger.debug("TOPICS - predicting query: " + str(query))
            # topics = tfidf_model.get_topics(
            #    topic_processing(query_text, bigrams), topn=5
            # )
            # logger.info(topics)
            # results["extracted"] = topics
        elif extractType == "summary":
            summary = GensimSumm(
                query_text, long_doc=False, word_count=30
            ).make_summary()
            results["extracted"] = summary
        elif extractType == "keywords":
            logger.debug("keywords - predicting query: " + str(query))
            results["extracted"] = get_keywords(query_text)

    except Exception:
        logger.error(f"Unable to get extract text for {query}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results


@router.post("/transSentenceSearch", status_code=200)
async def trans_sentence_infer(
    query: dict, response: Response, num_results: int = 5
) -> dict:
    """trans_sentence_infer - endpoint for sentence transformer inference
    Args:
        query: dict; format of query
            {"text": "i am text"}
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        results: dict; results of inference
    """
    logger.debug("SENTENCE TRANSFORMER - predicting query: " + str(query))
    results = {}
    try:
        query_text = query["text"]
        results = MODELS.sentence_trans.search(query_text, num_results)
        logger.info(results)
    except Exception:
        logger.error(
            f"Unable to get results from sentence transformer for {query}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results


@router.post("/questionAnswer", status_code=200)
async def qa_infer(query: dict, response: Response) -> dict:
    """qa_infer - endpoint for sentence transformer inference
    Args:
        query: dict; format of query, text must be concatenated string
            {"query": "what is the navy",
            "search_context":["pargraph 1", "xyz"]}
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        results: dict; results of inference
    """
    logger.debug("QUESTION ANSWER - predicting query: " + str(query["query"]))
    results = {}

    try:
        query_text = query["query"]
        query_context = query["search_context"]
        start = time.perf_counter()
        answers = MODELS.qa_model.answer(query_text, query_context)
        end = time.perf_counter()
        logger.info(answers)
        logger.info(f"time: {end - start:0.4f} seconds")
        results["answers"] = answers
        results["question"] = query_text

    except Exception:
        logger.error(f"Unable to get results from QA model for {query}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results


@router.post("/expandTerms", status_code=200)
async def post_expand_query_terms(termsList: dict, response: Response) -> dict:
    """post_expand_query_terms - endpoint for expand query terms
    Args:
        termsList: dict;
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        expansion_dict: dict; expanded dictionary of terms
    """

    terms_string = " ".join(termsList["termsList"])
    terms = preprocess(terms_string, remove_stopwords=True)
    expansion_dict = {}
    # logger.info("[{}] expanded: {}".format(user, termsList))

    logger.info(f"Expanding: {termsList}")
    try:
        for term in terms:
            term = unquoted(term)
            expansion_list = MODELS.query_expander.expand(
                term, **QexpConfig.MODEL_ARGS["expansion"]
            )
            # turn word pairs into search phrases since otherwise it will just search for pages with both words on them
            # removing original word from the return terms unless it is combined with another word
            logger.info(f"original expanded terms: {expansion_list}")
            finalTerms = remove_original_kw(expansion_list, term)
            expansion_dict[term] = ['"{}"'.format(exp) for exp in finalTerms]
            logger.info(f"-- Expanded {term} to \n {finalTerms}")
        terms = " ".join(terms)
        logger.info(f"Finding similiar words for: {terms}")
        sim_words_dict = MODELS.word_sim.most_similiar_tokens(terms)
        logger.info(f"-- Expanded {terms} to \n {sim_words_dict}")
        expanded_words = {}
        expanded_words["qexp"] = expansion_dict
        expanded_words["wordsim"] = sim_words_dict
        return expanded_words
    except Exception as e:
        logger.error(f"Error with query expansion on {termsList}")
        logger.error(e)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


@router.post("/wordSimilarity", status_code=200)
async def post_word_sim(termsDict: dict, response: Response) -> dict:
    """post_word_sim - endpoint for getting similar words
    Args:
        termsList: dict;
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        expansion_dict: dict; expanded dictionary of terms
    """
    # logger.info("[{}] expanded: {}".format(user, termsList))
    terms = termsDict["text"]
    logger.info(f"Finding similiar words for: {terms}")
    try:
        sim_words_dict = MODELS.word_sim.most_similiar_tokens(terms)
        logger.info(f"-- Expanded {terms} to \n {sim_words_dict}")
        return sim_words_dict
    except:
        logger.error(f"Error with query expansion on {terms}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR




@router.post("/transformerClassify", status_code=200)
async def transformer_classify(payload: list, response: Response) -> dict:
    """transformer_infer - endpoint for transformer inference
    Args:

    Returns:
        results: dict; results of inference
    """

    logger.info("TRANSFORMER - predicting text: " + str(payload))
    # TODO RAC update to table columns
    ## Update with actual db columns rather than these
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
    model_inputs_list = []
    try:
        for record in payload:
            # #how does this separate for each record?
            combined_text = ""
            # if 'budget_type' not in record:
            for concat_col in text_col_dict[record['budget_type']]:
                combined_text += f"{record.get(concat_col,'')} "
            model_inputs_list.append({"sentence":combined_text})
    except Exception as e:
        logger.error(f"Error parsing payload with exception: {e}")
        raise e

    classif_results_list = []
    ## Todo: should the batch size be a config, same for max_seq_len, if so where does it go?
    for classif_results in MODELS.classify_trans_jbook.predict(model_inputs_list, batch_size=16, max_seq_len=int(512)):
            classif_results_list += classif_results

    # extract out the "top class" for the record (numerically encoded prediction)
    classif_results_list = [classif_results['top_class'] for classif_results in classif_results_list]
    # map the numerically encoded
    classif_results_list = list(map(lambda classif_result: label_mapping[classif_result], classif_results_list))



    # construct return payload
    #
    #
    #     logger.info(results)
    # except Exception:
    #     logger.error(f"Unable to get results from transformer for {payload}")
    # response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    #     raise
    return classif_results_list


def unquoted(term):
    """unquoted - unquotes string
    Args:
        term: string
    Returns:
        term: without quotes
    """
    if term[0] in ["'", '"'] and term[-1] in ["'", '"']:
        return term[1:-1]
    else:
        return term

from fastapi import APIRouter, Response, status
import time
import requests
import hashlib

# must import sklearn first or you get an import error
from gamechangerml.src.search.query_expansion.utils import remove_original_kw
from gamechangerml.src.featurization.keywords.extract_keywords import get_keywords
from gamechangerml.src.text_handling.process import preprocess
from gamechangerml.api.fastapi.version import __version__

# from gamechangerml.models.topic_models.tfidf import bigrams, tfidf_model
# from gamechangerml.src.featurization.summary import GensimSumm
from gamechangerml.api.fastapi.settings import *
from gamechangerml.api.fastapi.model_loader import ModelLoader

from gamechangerml.configs.config import QexpConfig

router = APIRouter()
MODELS = ModelLoader()


@router.post("/transformerSearch", status_code=200)
async def transformer_infer(body: dict, response: Response) -> dict:
    """transformer_infer - endpoint for transformer inference
    Args:
        body: dict; json format of query
            {"query": "test", "documents": [{"text": "...", "id": "xxx"}, ...]
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        results: dict; results of inference
    """
    logger.debug("TRANSFORMER - predicting query: " + str(body))
    results = {}
    try:
        results = MODELS.sparse_reader.predict(body)
        logger.info(results)
    except Exception:
        logger.error(f"Unable to get results from transformer for {body}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results


@router.post("/textExtractions", status_code=200)
async def textExtract_infer(body: dict, extractType: str, response: Response) -> dict:
    """textExtract_infer - endpoint for sentence transformer inference
    Args:
        body: dict; json format of query
            {"text": "i am text"}
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
        extractType: url query string; one of topics, keywords, or summary
    Returns:
        results: dict; results of inference
    """
    results = {}
    try:
        query_text = body["text"]
        results["extractType"] = extractType
        if extractType == "topics":
            logger.debug("TOPICS - predicting query: " + str(body))
            topics = MODELS.topic_model.get_topics_from_text(query_text)
            logger.info(topics)
            results["extracted"] = topics
        elif extractType == "summary":
            # gensim upgrade breaks GensimSumm class
            # summary = GensimSumm(
            #     query_text, long_doc=False, word_count=30
            # ).make_summary()
            # results["extracted"] = summary
            results["extracted"] = "Summary is not supported at this time"
        elif extractType == "keywords":
            logger.debug("keywords - predicting query: " + str(body))
            results["extracted"] = get_keywords(query_text)

    except Exception:
        logger.error(f"Unable to get extract text for {body}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results


@router.post("/transSentenceSearch", status_code=200)
async def trans_sentence_infer(
    body: dict, response: Response, num_results: int = 10, externalSim: bool = False
) -> dict:
    """trans_sentence_infer - endpoint for sentence transformer inference
    Args:
        body: dict; json format of query
            {"text": "i am text"}
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        results: dict; results of inference
    """
    logger.debug("SENTENCE TRANSFORMER - predicting query: " + str(body))
    results = {}
    try:
        query_text = body["text"]
        results = MODELS.sentence_searcher.search(
            query_text, num_results, externalSim=False
        )
        logger.info(results)
    except Exception:
        logger.error(
            f"Unable to get results from sentence transformer for {body}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results


@router.post("/questionAnswer", status_code=200)
async def qa_infer(body: dict, response: Response) -> dict:
    """qa_infer - endpoint for sentence transformer inference
    Args:
        body: dict; json format of query, text must be concatenated string
            {"query": "what is the navy",
            "search_context":["pargraph 1", "xyz"]}
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        results: dict; results of inference
    """
    logger.debug("QUESTION ANSWER - predicting query: " + str(body["query"]))
    results = {}

    try:
        query_text = body["query"]
        query_context = body["search_context"]
        start = time.perf_counter()
        answers = MODELS.qa_model.answer(query_text, query_context)
        end = time.perf_counter()
        logger.info(answers)
        logger.info(f"time: {end - start:0.4f} seconds")
        results["answers"] = answers
        results["question"] = query_text

    except Exception:
        logger.error(f"Unable to get results from QA model for {body}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise
    return results


@router.post("/expandTerms", status_code=200)
async def post_expand_query_terms(body: dict, response: Response) -> dict:
    """post_expand_query_terms - endpoint for expand query terms
    Args:
        body: dict; json format of query
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        expansion_dict: dict; expanded dictionary of terms
    """

    terms_string = " ".join(body["termsList"])
    terms = preprocess(terms_string, remove_stopwords=True)
    expansion_dict = {}
    # logger.info("[{}] expanded: {}".format(user, termsList))

    logger.info(f"Expanding: {body}")
    query_expander = (
        MODELS.query_expander
        if body.get("qe_model", "gc_core") != "jbook"
        else MODELS.query_expander_jbook
    )
    try:
        for term in terms:
            term = unquoted(term)
            expansion_list = query_expander.expand(
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
        logger.error(f"Error with query expansion on {body}")
        logger.error(e)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


@router.post("/wordSimilarity", status_code=200)
async def post_word_sim(body: dict, response: Response) -> dict:
    """post_word_sim - endpoint for getting similar words
    Args:
        body: dict; json format of query
        Response: Response class; for status codes(apart of fastapi do not need to pass param)
    Returns:
        expansion_dict: dict; expanded dictionary of terms
    """
    # logger.info("[{}] expanded: {}".format(user, termsList))
    terms = body["text"]
    logger.info(f"Finding similiar words for: {terms}")
    try:
        sim_words_dict = MODELS.word_sim.most_similiar_tokens(terms)
        logger.info(f"-- Expanded {terms} to \n {sim_words_dict}")
        return sim_words_dict
    except:
        logger.error(f"Error with query expansion on {terms}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


@router.post("/recommender", status_code=200)
async def post_recommender(body: dict, response: Response) -> dict:
    results = {}
    sample = False
    try:
        filenames = body["filenames"]
        if not filenames:
            if body["sample"]:
                sample = body["sample"]
        logger.info(f"Recommending similar documents to {filenames}")
        results = MODELS.recommender.get_recs(
            filenames=filenames, sample=sample)
        if results['results'] != []:
            logger.info(f"Found similar docs: \n {str(results)}")
        else:
            logger.info("Did not find any similar docs")
    except Exception as e:
        logger.warning(f"Could not get similar docs for {filenames}")
        logger.warning(e)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return results


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

@router.get('/testExternal', status_code=200)
async def testExtRequest(response: Response) -> dict:
    token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6InN0ZXZlIiwiZGlzcGxheU5hbWUiOiJTZW5pb3IgU3RldmUiLCJwZXJtcyI6WyJWaWV3IEFnZW5jeSBTdXBwbHkgTWFuYWdlbWVudCIsIkRBUlEgYXJteSBSZXZpZXdlciA0IiwiVmlldyBBZ2VuY3kgbW9ja2RmYXMiLCJEQVJRIGRzcyBBZG1pbiIsIkJ1Y2tldGluZyBUaWNrZXQgUmVzb2x2ZSBGYWlsZWQgQ29ycmVjdGlvbiIsIkRBUlEgbW9ja2pvaW50c3RhZmYgUmV2aWV3ZXIgMSIsIlZpZXcgQWdlbmN5IFRTTyIsIlZpZXcgQWdlbmN5IGRvZGVhIiwiREFSUSBkZmFzIFRlc3RlciIsIkRBUlEgZHRyYSBBZG1pbiIsIkRBUlEgZHRpYyBVc2VyIiwiVmlldyBBZ2VuY3kgbW9ja2R0cmEiLCJCdWNrZXRpbmcgVGlja2V0IEFzc2lnbiIsIkRBUlEgbmF2eSBDb29yZGluYXRvciAzIiwiREFSUSBtb2NrZHNjYSBSZXZpZXdlciAzIiwiVmlldyBBZ2VuY3kgRFNTIiwiVmlldyBBZ2VuY3kgRE9KIiwiREFSUSBkaHAgQ29vcmRpbmF0b3IgMyIsIlZpZXcgQWdlbmN5IERBVSIsIkRBUlEgd2hzIFRlc3RlciIsIkRBUlEgbW9ja21kYSBDb29yZGluYXRvciAxIiwiREFSUSBkY2FhIFJldmlld2VyIDIiLCJEQVJRIG1vY2t3aHMgUmV2aWV3ZXIgMSIsIkRBUlEgZGlzYSBDb29yZGluYXRvciAyIiwiREFSUSBtb2Nrd2hzIENvb3JkaW5hdG9yIDEiLCJEQVJRIG1vY2tkdHJhIFRlc3RlciIsIkRBUlEgbW9ja2RocCBDb29yZGluYXRvciAzIiwiVmlldyBBZ2VuY3kgTmF2eSAvIE1DIFNoYXJlZCIsIkRBUlEgbW9ja3VzbWMgR3JvdXAgQWRtaW4iLCJEQVJRIHNvY29tIFVzZXIgQWRtaW4iLCJEQVJRIGFpcmZvcmNlIENvb3JkaW5hdG9yIDEiLCJEQVJRIGRzY2EgVXNlciIsIkRBUlEgYXJteSBUZXN0ZXIiLCJEQVJRIGFybXkgSFEiLCJWaWV3IEFnZW5jeSBET1QiLCJEQVJRIG1vY2t3aHMgQWRtaW4iLCJWaWV3IEFnZW5jeSBUcmFuc3BvcnRhdGlvbiIsIkRBUlEgZHRzYSBBZG1pbiIsIkRBUlEgbWRhIFJldmlld2VyIDMiLCJWaWV3IEFnZW5jeSBDb21wb25lbnQgTGV2ZWwiLCJWaWV3IEFnZW5jeSBDQk8iLCJEQVJRIGNiZHAgUmV2aWV3ZXIgMiIsIkRBUlEgYWdlbmN5eCBVc2VyIiwiVmlldyBBZ2VuY3kgVVMgU3BlY2lhbCBPcGVyYXRpb25zIENvbW1hbmQiLCJEQVJRIGRtZWEgQWRtaW4iLCJWaWV3IEFnZW5jeSBPRUEiLCJEQVJRIG1vY2thcm15IFZpZXcgT25seSIsIkRBUlEgZGhwIFJldmlld2VyIDEiLCJEQVJRIG1vY2tuYXZ5IENvb3JkaW5hdG9yIDEiLCJWaWV3IEFnZW5jeSBOYXZ5IEdlbmVyYWwgRnVuZCIsIlZpZXcgQWdlbmN5IGFnZW5jeSBhIiwiVmlldyBBZ2VuY3kgTEkiLCJWaWV3IEFnZW5jeSBDTVRQQyIsIkRBUlEgbW9ja2RhcnBhIFRlc3RlciIsIlZpZXcgQWdlbmN5IG1vY2tkc2NhIiwiREFSUSBtb2NrbmF2eSBUZXN0ZXIiLCJEQVJRIGRjYWEgVmlldyBPbmx5IiwiVmlldyBERkFTIFJlY29uIiwiREFSUSBhaXJmb3JjZSBSZXZpZXdlciAzIiwiREFSUSBtb2NrYXJteSBVc2VyIiwiREFSUSBkY2FhIFJldmlld2VyIDEiLCJWaWV3IEFnZW5jeSBBRklTIiwiVmlldyBBZ2VuY3kgc29jb20iLCJEQVJRIGFnZW5jeXggUmV2aWV3ZXIgNCIsIkRBUlEgd2hzIEFkbWluIiwiQnVsayBHZW5lcmF0ZSIsIlZpZXcgQWdlbmN5IERPTCIsIkRBUlEgd2hzIFVzZXIiLCJWaWV3IEFnZW5jeSBBcm15IEdlbmVyYWwgRnVuZCIsIlZpZXcgQWdlbmN5IGFnZW5jeXgiLCJEQVJRIGRtZWEgVmlldyBPbmx5IiwiREFSUSBkb2RlYSBBZG1pbiIsIkRBUlEgbW9ja2FybXkgUmV2aWV3ZXIgNSIsIlZpZXcgQWdlbmN5IERPREVBIiwiVmlldyBBZ2VuY3kgRFRSQSIsIlZpZXcgQWdlbmN5IE9TRCIsIkRBUlEgZHRzYSBWaWV3IE9ubHkiLCJEQVJRIG1vY2t1c21jIENvb3JkaW5hdG9yIDIiLCJEQVJRIGRocCBSZXZpZXdlciAyIiwiREFSUSBkc2NhIFVzZXIgQWRtaW4iLCJEQVJRIG1vY2tqb2ludHN0YWZmIFJldmlld2VyIDIiLCJEQVJRIG1kYSBSZXZpZXdlciAyIiwiT3RoZXIgRGVmZW5zZSBBY3Rpdml0aWVzIC0gVGllciAzICYgNCIsIkRBUlEgYWdlbmN5eCBHcm91cCBBZG1pbiIsIkRBUlEgZGNhYSBDb29yZGluYXRvciAxIiwiREFSUSBkcGFhIFVzZXIiLCJWaWV3IEFnZW5jeSBkcGFhIiwiVmlldyBBZ2VuY3kgbW9ja2RocCIsIkRBUlEgbW9ja25hdnkgVXNlciBBZG1pbiIsIkRBUlEgbW9ja3VzbWMgQ29vcmRpbmF0b3IgMSIsIlZpZXcgQWdlbmN5IE90aGVyIERlZmVuc2UgQWN0aXZpdGllcyAtIFRpZXIgMiIsIkRBUlEgbWRhIFJldmlld2VyIDEiLCJEQVJRIGFnZW5jeXggUmV2aWV3ZXIgMyIsIkRBUlEgZGNtYSBUZXN0ZXIiLCJEQVJRIGRjbWEgUmV2aWV3ZXIgMSIsIkRBUlEgbW9ja21kYSBVc2VyIiwiVmlldyBBZ2VuY3kgVVNNQyIsIlZpZXcgQWdlbmN5IG1vY2tkaHB1c3VocyIsIkRBUlEgZGlzYSBVc2VyIiwiVmlldyBBZ2VuY3kgZGFycGEiLCJVcGRhdGUgRGF0YSBTdGF0dXMgVHJhY2tlciIsIlZpZXcgQWdlbmN5IGpvaW50c3RhZmYiLCJWaWV3IEFnZW5jeSBkb2RpZyIsIlZpZXcgQWdlbmN5IG1vY2t3aHMiLCJEQVJRIG9lYSBVc2VyIiwiREFSUSBtb2NrZHRyYSBIUSIsIkRBUlEgY2JkcCBBZG1pbiIsIlZpZXcgQWdlbmN5IEdTQSIsIlZpZXcgQWdlbmN5IFVTQSIsIkRBUlEgZHRpYyBBZG1pbiIsImNhdGFsb2dNZXRhZGF0YVNlYXJjaEF1dG9Db21wbGV0ZVBPU1QiLCJWaWV3IEFnZW5jeSBCYXNlIFN1cHBvcnQiLCJWaWV3IEFnZW5jeSB1c21jIiwiVmlldyBBZ2VuY3kgUk8iLCJEQVJRIGRzY2EgUmV2aWV3ZXIgMSIsIkRBUlEgZG9kZWEgVmlldyBPbmx5IiwiREFSUSBkaHAgVGVzdGVyIiwiVmlldyBBZ2VuY3kgVVNQSFMiLCJEQVJRIG1vY2tkZmFzIENvb3JkaW5hdG9yIDIiLCJEQVJRIG1vY2tqb2ludHN0YWZmIFVzZXIiLCJEQVJRIG1vY2tkaHAgQ29vcmRpbmF0b3IgMiIsIlZpZXcgQWdlbmN5IGNiZHAiLCJWaWV3IEFnZW5jeSBJbmR1c3RyaWFsIE9wZXJhdGlvbnMiLCJCdWNrZXRpbmcgVGlja2V0IE1vdmUgUmVqZWN0ZWQiLCJWaWV3IEFnZW5jeSBWQUNBIiwiREFSUSBtb2NrbmF2eSBWaWV3IE9ubHkiLCJEQVJRIGRpc2EgUmV2aWV3ZXIgMiIsIlZpZXcgRGFzaGJvYXJkIiwiREFSUSBkaXNhIEdyb3VwIEFkbWluIiwiVmlldyBBZ2VuY3kgVVNUUkFOU0NPTSIsIkRBUlEgbW9ja21kYSBDb29yZGluYXRvciAyIiwiVmlldyBBZ2VuY3kgRFBNTyIsIkRBUlEgbmF2eSBBZG1pbiIsIlZpZXcgRGF0YXBvcnRhbCIsIkRBUlEgbmF2eSBVc2VyIiwiVmlldyBBZ2VuY3kgRE9JIiwiVmlldyBBZ2VuY3kgZG90ZSIsIkRBUlEgbW9ja2pvaW50c3RhZmYgQWRtaW4iLCJEQVJRIGRocmEgVmlldyBPbmx5IiwiREFSUSBtb2NrbmF2eSBVc2VyIiwiREFSUSBtb2NrZHNjYSBDb29yZGluYXRvciAxIiwiREFSUSBzb2NvbSBWaWV3IE9ubHkiLCJEQVJRIGNiZHAgVXNlciIsIkRBUlEgbmF2eSBUZXN0ZXIiLCJWaWV3IEFnZW5jeSBEQ01BIiwiREFSUSBkZmFzIFZpZXcgT25seSIsIkRBUlEgbW9ja3docyBVc2VyIiwiUHVibGlzaCBEYXRhc2V0IiwiREFSUSBkZmFzIENvb3JkaW5hdG9yIDEiLCJWaWV3IEFnZW5jeSBHQU8iLCJEQVJRIGNiZHAgVGVzdGVyIiwiVmlldyBHdWlkZWQgUXVlcnkiLCJWaWV3IEFnZW5jeSBBQUZFUyIsIkRBUlEgZGF1IEFkbWluIiwiVmlldyBBZ2VuY3kgVEVTVCIsIlZpZXcgQWdlbmN5IFRNQSIsIlZpZXcgQWdlbmN5IEpQIiwiREFSUSBkaHJhIFVzZXIiLCJEQVJRIG1vY2t1c21jIFJldmlld2VyIDUiLCJEQVJRIGFybXkgQ29vcmRpbmF0b3IgMyIsIldlYmFwcCBBZG1pbiIsIkRBUlEgZGNtYSBBZG1pbiIsIlZpZXcgQWdlbmN5IEFjdGl2ZSBBcm15IiwiVmlldyBBZ2VuY3kgT0ZGSUNFIE9GIFRIRSBJTlNQRUNUT1IgR0VORVJBTCIsIkRBUlEgbW9ja2pvaW50c3RhZmYgVGVzdGVyIiwiRWRpdCBEYXNoYm9hcmQiLCJEQVJRIG9lYSBWaWV3IE9ubHkiLCJEQVJRIGRhdSBDb29yZGluYXRvciAxIiwiREFSUSBjYmRwIENvb3JkaW5hdG9yIDEiLCJEQVJRIGRocCBHcm91cCBBZG1pbiIsIkRBUlEgbW9ja2RmYXMgR3JvdXAgQWRtaW4iLCJDcmVhdGUgRGF0YXNldCIsIkRBUlEgbW9ja25hdnkgUmV2aWV3ZXIgNSIsIkRBUlEgbW9ja2FybXkgQ29vcmRpbmF0b3IgMyIsIkRBUlEgZGNhYSBBZG1pbiIsIkRBUlEgZG9kZWEgVXNlciIsIlZpZXcgQWdlbmN5IERJQSIsIkRBUlEgbW9ja25hdnkgUmV2aWV3ZXIgMSIsIkRBUlEgbW9ja21kYSBSZXZpZXdlciAzIiwiVmlldyBBZ2VuY3kgRENBQSIsIkRBUlEgbW9ja3VzbWMgQWRtaW4iLCJEQVJRIGRvZGlnIFJldmlld2VyIDEiLCJWaWV3IEFnZW5jeSBhZ2VuY3kgYiIsIkRBUlEgbW9ja2RocCBSZXZpZXdlciAyIiwiVmlldyBBZ2VuY3kgREhTIiwiQnVja2V0aW5nIFRpY2tldCBNb3ZlIE5lZWRzIExvZ2ljIENoYW5nZSIsIkRBUlEgbW9ja3docyBSZXZpZXdlciAyIiwiREFSUSBtb2NrZGZhcyBVc2VyIEFkbWluIiwiRGVsZXRlIFdvcmtib29rcyIsIlZpZXcgQWdlbmN5IEhVRCIsIlZpZXcgQWdlbmN5IElOIiwiREFSUSBkYXJwYSBSZXZpZXdlciAxIiwiVmlldyBWYXJpYW5jZSIsIlZpZXcgQWdlbmN5IGRocCIsIkRBUlEgbW9ja2RocCBVc2VyIiwiREFSUSBtb2NrZGhwIFVzZXIgQWRtaW4iLCJEQVJRIGRvZGlnIFZpZXcgT25seSIsIkRBUlEgZGxhIEFkbWluIiwiREFSUSBhcm15IFJldmlld2VyIDUiLCJWaWV3IEFnZW5jeSBET0MiLCJEQVJRIGpvaW50c3RhZmYgVmlldyBPbmx5IiwiREFSUSBhZ2VuY3l4IFRlc3RlciIsIkRBUlEgbW9ja2RmYXMgUmV2aWV3ZXIgMiIsIkRBUlEgbW9ja25hdnkgUmV2aWV3ZXIgNCIsIkRBUlEgZGVjYSBBZG1pbiIsIlZpZXcgQWdlbmN5IERDTVNBIiwiREFSUSBkYXJwYSBWaWV3IE9ubHkiLCJEQVJRIG1vY2thcm15IEFkbWluIiwiVmlldyBBZ2VuY3kgQXJteSBOYXRpb25hbCBHdWFyZCIsIlZpZXcgQWdlbmN5IFVTVUhTIiwiVmlldyBBZ2VuY3kgZGlzYSIsIkRBUlEgbWRhIFZpZXcgT25seSIsIkRBUlEgZHRzYSBVc2VyIiwiREFSUSBkbWEgQWRtaW4iLCJEQVJRIG1kYSBDb29yZGluYXRvciAyIiwiVmlldyBBZ2VuY3kgZGxhIiwiVmlldyBBZ2VuY3kgZGZhcyIsIkRBUlEgZGFycGEgQWRtaW4iLCJEQVJRIG1vY2tkZmFzIFRlc3RlciIsIkRBUlEgZGNhYSBVc2VyIiwiREFSUSBkY21hIFJldmlld2VyIDQiLCJWaWV3IEFnZW5jeSBkc2NhIiwiVmlldyBBZ2VuY3kgREZBUyIsIlZpZXcgQWdlbmN5IE1EQSIsIlZpZXcgQWdlbmN5IERUSUMiLCJWaWV3IFVzZXIgTWFuYWdlbWVudCIsIkRBUlEgbmF2eSBVc2VyIEFkbWluIiwiREFSUSBtb2NrbWRhIFZpZXcgT25seSIsIkRBUlEgYXJteSBDb29yZGluYXRvciAyIiwiREFSUSBkb2RpZyBDb29yZGluYXRvciAxIiwiREFSUSBqb2ludHN0YWZmIFJldmlld2VyIDEiLCJEQVJRIG5hdnkgUmV2aWV3ZXIgNCIsIlZpZXcgTmF2eSBEYXRhIiwiYmlnZGF0YSBqb2luIiwiREFSUSBkb3RlIFZpZXcgT25seSIsIkRBUlEgYWdlbmN5eCBDb29yZGluYXRvciIsImNhdGFsb2dNZXRhZGF0YVNlYXJjaFBPU1QiLCJWaWV3IEFnZW5jeSBXSFMiLCJWaWV3IEFnZW5jeSBkdGljIiwiREFSUSBtb2NrZHRyYSBWaWV3IE9ubHkiLCJCdWNrZXRpbmcgVGlja2V0IE1vdmUgUGVuZGluZyBDb3JyZWN0aW9uIiwiVmlldyBBZ2VuY3kgRE9TIiwiREFSUSBtb2NrZGFycGEgVXNlciIsIkRBUlEgbW9ja2FybXkgQ29vcmRpbmF0b3IgMiIsIlZpZXcgQWdlbmN5IFVTRChBJlQpRFNBIiwiVmlldyBBZ2VuY3kgQWlyIEZvcmNlIFJlc2VydmUiLCJEQVJRIG5hdnkgUmV2aWV3ZXIgMyIsIlZpZXcgQWdlbmN5IG1vY2tqb2ludHN0YWZmIiwiREFSUSBtb2NrZGZhcyBIUSIsIlZpZXcgSERGUyBSb290IiwiVmlldyBBZ2VuY3kgU2VuYXRlIiwiVmlldyBBZ2VuY3kgZHNzIiwiREFSUSBTZWxmIEFzc2lnbiIsIkRBUlEgZHRpYyBVc2VyIEFkbWluIiwiREFSUSBkc2NhIFJldmlld2VyIDIiLCJEQVJRIGRzY2EgR3JvdXAgQWRtaW4iLCJEQVJRIHdocyBDb29yZGluYXRvciAxIiwiREFSUSBkc2NhIFZpZXcgT25seSIsIlZpZXcgQWdlbmN5IE9QTSIsIkRBUlEgYWlyZm9yY2UgUmV2aWV3ZXIgMiIsIkRBUlEgZHRyYSBIUSIsIlZpZXcgQWdlbmN5IGRocC11c3VocyIsIkRBUlEgbWRhIENvb3JkaW5hdG9yIDMiLCJWaWV3IEFnZW5jeSBVU0FGIiwiVmlldyBBZ2VuY3kgZGhwdXN1aHMiLCJEQVJRIGFybXkgQWRtaW4iLCJEQVJRIGR0cmEgVXNlciBBZG1pbiIsIkRBUlEgZGZhcyBSZXZpZXdlciAzIiwiVmlldyBBZ2VuY3kgbW9ja21kYSIsIlZpZXcgQWdlbmN5IFRSRUFTIERFUFQiLCJEQVJRIG1vY2ttZGEgQWRtaW4iLCJEQVJRIG1vY2tkdHJhIFJldmlld2VyIDIiLCJTdWJtaXQgV29ya2Jvb2tzIiwiREFSUSBhaXJmb3JjZSBSZXZpZXdlciA1IiwiREFSUSBkc2NhIENvb3JkaW5hdG9yIDEiLCJWaWV3IEFnZW5jeSB1c3NvY29tIiwiREFSUSBhZ2VuY3l4IEFkbWluIiwiVmlldyBBZ2VuY3kgZG1lYSIsIlZpZXcgQWdlbmN5IE5EVSIsIlZpZXcgTWFnZWxsYW4iLCJWaWV3IEFnZW5jeSBkY21hIiwiREFSUSBtb2NrZHNjYSBIUSIsIlZpZXcgQWdlbmN5IGFpcmZvcmNlIiwiREFSUSBhZ2VuY3l4IFJldmlld2VyIDEiLCJWaWV3IEFnZW5jeSBGQkkiLCJWaWV3IEFnZW5jeSBQRlBBIiwiREFSUSBtb2NrdXNtYyBUZXN0ZXIiLCJWaWV3IEFnZW5jeSBNYXJpbmUgQ29ycHMgLSBOV0NGIiwiREFSUSBkdHJhIFZpZXcgT25seSIsIkRBUlEgZGZhcyBVc2VyIiwiREFSUSBtb2NrZHRyYSBSZXZpZXdlciAxIiwiVmlldyBBZ2VuY3kgRE1BIiwiREFSUSBtb2NrYXJteSBSZXZpZXdlciAxIiwiREFSUSBjYmRwIFJldmlld2VyIDEiLCJEQVJRIG1vY2t1c21jIFJldmlld2VyIDMiLCJWaWV3IEFnZW5jeSBQUlBPIiwiREFSUSBtZGEgUmV2aWV3ZXIgNCIsIkRBUlEgYWlyZm9yY2UgUmV2aWV3ZXIgNCIsIkRBUlEgbW9ja3VzbWMgUmV2aWV3ZXIgMSIsIkRBUlEgZGlzYSBSZXZpZXdlciAzIiwiREFSUSBkbWEgVmlldyBPbmx5IiwiREFSUSBtb2NrZGhwIFJldmlld2VyIDEiLCJEQVJRIG1vY2tkZmFzIEFkbWluIiwiQWRtaW5pc3RlciBPU0QgVGlja21hcmtzIiwiREFSUSBuYXZ5IENvb3JkaW5hdG9yIDIiLCJEQVJRIGpvaW50c3RhZmYgUmV2aWV3ZXIgMiIsIlZpZXcgQWdlbmN5IG1kYSIsIkRBUlEgbW9ja21kYSBSZXZpZXdlciAxIiwiREFSUSBkaHAgQ29vcmRpbmF0b3IgMSIsIlZpZXcgQWdlbmN5IG9lYSIsIkRBUlEgZGZhcyBSZXZpZXdlciAxIiwiVmlldyBBZ2VuY3kgbW9ja2FybXkiLCJWaWV3IEFnZW5jeSBTQ1VTIiwiREFSUSBtb2NrZGFycGEgQ29vcmRpbmF0b3IgMSIsIkRBUlEgYXJteSBHcm91cCBBZG1pbiIsIkRBUlEgYXJteSBSZXZpZXdlciAzIiwiVmlldyBBZ2VuY3kgZGhhIiwiVmlldyBBZ2VuY3kgTlNBIiwiVmlldyBBZ2VuY3kgVVNOIiwiREFSUSBkYXJwYSBIUSIsIkRBUlEgbW9ja2RzY2EgUmV2aWV3ZXIgMSIsIkRBUlEgbW9ja2FybXkgVGVzdGVyIiwiVmlldyBBZ2VuY3kgVkEiLCJEQVJRIG1vY2tkaHAgVmlldyBPbmx5IiwiVmlldyBBZ2VuY3kgU2VsZWN0IEFsbCBTdWJlbnRpdGllcyIsIkRBUlEgZGFycGEgQ29vcmRpbmF0b3IgMSIsIkRBUlEgbW9ja3VzbWMgSFEiLCJWaWV3IEFnZW5jeSBkdHJhIiwiREFSUSBkb2RpZyBVc2VyIiwiREFSUSBtb2NrZGhwIEFkbWluIiwiREFSUSBtb2Nram9pbnRzdGFmZiBDb29yZGluYXRvciAyIiwiREFSUSBtb2NrZHRyYSBHcm91cCBBZG1pbiIsIkRBUlEgbW9ja2RzY2EgVXNlciIsIkRBUlEgZHNjYSBDb29yZGluYXRvciAyIiwiREFSUSBkaXNhIFJldmlld2VyIDEiLCJEQVJRIG1vY2thcm15IFJldmlld2VyIDIiLCJEQVJRIG1vY2tkZmFzIENvb3JkaW5hdG9yIDEiLCJWaWV3IEFnZW5jeSBFREEiLCJWaWV3IEFnZW5jeSBtb2NrZGFycGEiLCJJbXBvcnQgVGlja21hcmtzIiwiVmlldyBBZ2VuY3kgVVNDT0EiLCJEQVJRIG1vY2tkdHJhIFVzZXIgQWRtaW4iLCJWaWV3IEFnZW5jeSBEb0VEIiwiREFSUSBtb2NrZGhwIENvb3JkaW5hdG9yIDEiLCJEQVJRIGFybXkgVXNlciBBZG1pbiIsIkRBUlEgbW9ja2R0cmEgQWRtaW4iLCJEQVJRIG1vY2tkYXJwYSBIUSIsIkRBUlEgbW9ja21kYSBSZXZpZXdlciAyIiwiREFSUSBkaHAgQWRtaW4iLCJEQVJRIGpvaW50c3RhZmYgVXNlciIsIkRBUlEgbW9ja2FybXkgVXNlciBBZG1pbiIsIkRBUlEgZGNhYSBUZXN0ZXIiLCJWaWV3IEFnZW5jeSBVU1BTIiwiREFSUSBkc3MgVmlldyBPbmx5IiwiREFSUSBkbWEgVXNlciIsIkRBUlEgZGFycGEgVGVzdGVyIiwiREFSUSBhcm15IFZpZXcgT25seSIsIkRBUlEgZHNzIFVzZXIgQWRtaW4iLCJBY2Nlc3MgRGF0YSBBUEkiLCJEQVJRIGRvdGUgQWRtaW4iLCJEQVJRIGR0cmEgVXNlciIsIkRBUlEgam9pbnRzdGFmZiBDb29yZGluYXRvciAxIiwiVmlldyBEYXRhc2V0IE1hbmFnZW1lbnQiLCJGU0QgV29ya2Jvb2sgUHVibGlzaCBBaXIgRm9yY2UiLCJEQVJRIG1vY2tuYXZ5IENvb3JkaW5hdG9yIDIiLCJEQVJRIG5hdnkgQ29vcmRpbmF0b3IgMSIsIkRBUlEgZGlzYSBBZG1pbiIsIkRBUlEgd2hzIFZpZXcgT25seSIsIkRBUlEgZGZhcyBVc2VyIEFkbWluIiwiVmlldyBBZ2VuY3kgTkFSQSIsIlZpZXcgQWdlbmN5IFRSTUMiLCJCdWNrZXRpbmcgVGlja2V0IFJlc29sdmUgUGVuZGluZyBBcHByb3ZhbCIsIlZpZXcgQWdlbmN5IEFybXkgV29ya2luZyBDYXBpdGFsIEZ1bmQiLCJWaWV3IEFnZW5jeSBDTyIsIkRBUlEgbWRhIFRlc3RlciIsIlZpZXcgQWdlbmN5IE1hcmluZSBDb3JwcyIsIkRBUlEgYXJteSBSZXZpZXdlciAxIiwiREFSUSBhaXJmb3JjZSBDb29yZGluYXRvciAzIiwiVmlldyBBbGwgQWdlbmNpZXMiLCJEQVJRIG1vY2tkYXJwYSBSZXZpZXdlciAxIiwiREFSUSBtb2NrZHNjYSBUZXN0ZXIiLCJEQVJRIGRvdGUgVXNlciIsIlZpZXcgQWdlbmN5IERMU0EiLCJCdWNrZXRpbmcgVGlja2V0IE1vdmUgUGVuZGluZyBMb2dpYyIsIlZpZXcgQWdlbmN5IEhIUyIsIkRBUlEgZGNtYSBSZXZpZXdlciAyIiwiREFSUSBtb2Nrd2hzIEhRIiwiREFSUSBtb2NrYXJteSBSZXZpZXdlciAzIiwiVmlldyBBZ2VuY3kgVVNDRyIsIkRBUlEgYWlyZm9yY2UgQ29vcmRpbmF0b3IgMiIsIkRBUlEgYWlyZm9yY2UgVGVzdGVyIiwiREFSUSBhZ2VuY3l4IFJldmlld2VyIDIiLCJEQVJRIG1vY2thcm15IFJldmlld2VyIDQiLCJDYW4gRXhwb3J0IEZyb20gREFSUSIsIkRBUlEgZGNtYSBSZXZpZXdlciAzIiwiREFSUSBtb2NrdXNtYyBSZXZpZXdlciAyIiwiVmlldyBBZ2VuY3kgTmF2eSBXb3JraW5nIENhcGl0YWwgRnVuZCIsIkRBUlEgZGhyYSBBZG1pbiIsIkRBUlEgbW9ja2RzY2EgUmV2aWV3ZXIgMiIsIkRBUlEgZHRpYyBWaWV3IE9ubHkiLCJEQVJRIG1vY2t1c21jIFZpZXcgT25seSIsIkJ1Y2tldGluZyBUaWNrZXQgUmVvcGVuIiwiREFSUSBtZGEgQ29vcmRpbmF0b3IgMSIsIlZpZXcgQWdlbmN5IFJlcHJlc2VudGF0aXZlcyIsIkRBUlEgd2hzIFJldmlld2VyIDEiLCJEQVJRIG1vY2tuYXZ5IEhRIiwiVmlldyBEYXRhYmFzZSBRdWVyeSIsIkRBUlEgYWdlbmN5eCBDb29yZGluYXRvciAzIiwiREFSUSBkYXJwYSBVc2VyIiwiREFSUSB1c21jIFVzZXIiLCJWaWV3IEFnZW5jeSBEQVJQQSIsIkRBUlEgbW9ja25hdnkgR3JvdXAgQWRtaW4iLCJEQVJRIGR0cmEgQ29vcmRpbmF0b3IgMSIsIkRBUlEgY2JkcCBWaWV3IE9ubHkiLCJXZWJhcHAgU3VwZXIgQWRtaW4iLCJEQVJRIG1vY2tuYXZ5IFJldmlld2VyIDIiLCJEQVJRIGRhcnBhIENvb3JkaW5hdG9yIDIiLCJWaWV3IEFnZW5jeSBkaHJhIiwiVmlldyBBZ2VuY3kgRVJTQyIsIlZpZXcgQWdlbmN5IE90aGVyIiwiQnVja2V0aW5nIFRpY2tldCBSZXNvbHZlIEluIFByb2dyZXNzIiwiREFSUSBtb2NrdXNtYyBVc2VyIiwiREFSUSBkaGEgQWRtaW4iLCJEQVJRIG1vY2tkZmFzIFJldmlld2VyIDMiLCJEQVJRIG1vY2tuYXZ5IENvb3JkaW5hdG9yIDMiLCJEQVJRIG1vY2tkdHJhIFVzZXIiLCJEQVJRIGRpc2EgVmlldyBPbmx5IiwiREFSUSBkaXNhIFRlc3RlciIsIkRBUlEgZHBhYSBWaWV3IE9ubHkiLCJWaWV3IEFnZW5jeSBDQVBTVyIsIkJ1bGwgQXJlbmEgQWNjZXNzIiwiVmlldyBBZ2VuY3kgQWlyIEZvcmNlIFdvcmtpbmcgQ2FwaXRhbCBGdW5kIiwiREFSUSBtb2NrZGFycGEgUmV2aWV3ZXIgMyIsIkRBUlEgc29jb20gR3JvdXAgQWRtaW4iLCJQdWJsaXNoIFdvcmtib29rcyIsIkRBUlEgZGhhIFZpZXcgT25seSIsIlZpZXcgQWdlbmN5IGFybXkiLCJEQVJRIG1vY2t3aHMgUmV2aWV3ZXIgNSIsIlZpZXcgQWdlbmN5IERlcGFydG1lbnQgb2YgQWlyIEZvcmNlLCBXb3JraW5nIENhcGl0YWwgRnVuZCIsIkRBUlEgZG1lYSBVc2VyIiwiVmlldyBBZ2VuY3kgbmF2eSIsIlZpZXcgQWdlbmN5IEFybXkgUmVzZXJ2ZSIsIkRBUlEgbmF2eSBSZXZpZXdlciA1IiwiVmlldyBBZ2VuY3kgTU9DQVMiLCJEQVJRIG1vY2tkZmFzIFZpZXcgT25seSIsIkRBUlEgbW9ja3VzbWMgUmV2aWV3ZXIgNCIsIkRBUlEgbW9ja2FybXkgSFEiLCJEQVJRIG5hdnkgR3JvdXAgQWRtaW4iLCJEQVJRIGRzcyBHcm91cCBBZG1pbiIsIlZpZXcgQWdlbmN5IERJU0EiLCJWaWV3IEFnZW5jeSBPTkRDUCIsIkRBUlEgbW9ja21kYSBSZXZpZXdlciA0IiwiREFSUSBqb2ludHN0YWZmIFRlc3RlciIsIkRBUlEgbW9ja2RzY2EgVmlldyBPbmx5IiwiREFSUSBhZ2VuY3l4IFZpZXcgT25seSIsIkRBUlEgZHRpYyBHcm91cCBBZG1pbiIsIkRBUlEgZGxhIFZpZXcgT25seSIsIkRBUlEgZGlzYSBVc2VyIEFkbWluIiwiREFSUSBkbGEgVXNlciIsIlZpZXcgQWdlbmN5IGFnZW5jeSBjIiwiREFSUSBuYXZ5IFZpZXcgT25seSIsIkRBUlEgYXJteSBDb29yZGluYXRvciAxIiwiREFSUSBkaHAgQ29vcmRpbmF0b3IgMiIsIkRBUlEgbW9ja3VzbWMgQ29vcmRpbmF0b3IgMyIsIkRBUlEgYWlyZm9yY2UgUmV2aWV3ZXIgMSIsIlZpZXcgQWdlbmN5IE5BU0EiLCJEQVJRIGRzY2EgVGVzdGVyIiwiREFSUSBzb2NvbSBVc2VyIiwiVmlldyBBZ2VuY3kgRFNDQSIsIkRBUlEgbW9ja2RocCBHcm91cCBBZG1pbiIsIlZpZXcgQWdlbmN5IHdocyIsIkRBUlEgZGhwIFVzZXIgQWRtaW4iLCJWaWV3IEFnZW5jeSBET0UiLCJWaWV3IEFnZW5jeSBkdHNhIiwiREFSUSBtb2Nrd2hzIFJldmlld2VyIDMiLCJEQVJRIG1vY2tkYXJwYSBBZG1pbiIsIkRBUlEgbW9ja2FybXkgR3JvdXAgQWRtaW4iLCJCdWNrZXRpbmcgVGlja2V0IE1vdmUgTG9naWMgVXBkYXRlZCIsIkRBUlEgZHNjYSBSZXZpZXdlciAzIiwiREFSUSBkaHAgVmlldyBPbmx5IiwiREFSUSBtb2NrZHNjYSBDb29yZGluYXRvciAyIiwiVmlldyBBZ2VuY3kgREVBTVMiLCJEQVJRIGpvaW50c3RhZmYgQ29vcmRpbmF0b3IgMiIsIkFkbWluaXN0ZXIgTmF2eSBUaWNrbWFya3MiLCJWaWV3IEFnZW5jeSBBaXIgRm9yY2UgQWN0aXZlIiwiVmlldyBBZ2VuY3kgRG9EIiwiREFSUSBtb2Nram9pbnRzdGFmZiBDb29yZGluYXRvciAxIiwiVmlldyBBZ2VuY3kgbW9ja3VzbWMiLCJWaWV3IEFnZW5jeSBhaXIgZm9yY2UiLCJEQVJRIGRmYXMgQWRtaW4iLCJWaWV3IEFnZW5jeSBOT0FBIiwiVmlldyBBZ2VuY3kgU09DT00iLCJEQVJRIGRvZGlnIFRlc3RlciIsIlZpZXcgQWdlbmN5IGRjYWEiLCJEQVJRIGRzY2EgQWRtaW4iLCJEQVJRIGRocCBVc2VyIiwiREFSUSBtb2Nrd2hzIFZpZXcgT25seSIsIkRBUlEgZGNtYSBDb29yZGluYXRvciAxIiwiREFSUSBkZmFzIFJldmlld2VyIDIiLCJEQVJRIG1vY2tkdHJhIENvb3JkaW5hdG9yIDEiLCJEQVJRIGFnZW5jeXggQ29vcmRpbmF0b3IgMSIsIkRBUlEgbW9ja3docyBSZXZpZXdlciA0IiwiVmlldyBBZ2VuY3kgZGF1IiwiREFSUSBkYXJwYSBSZXZpZXdlciAyIiwiREFSUSBtb2Nrd2hzIFRlc3RlciIsIlZpZXcgQWdlbmN5IEVPUE9WUCIsIkRBUlEgZGFwcmEgUmV2aWV3ZXIgMSIsIlZpZXcgQWdlbmN5IGRtYSIsIlZpZXcgQWdlbmN5IERMQSIsIkRBUlEgb2VhIEFkbWluIiwiREFSUSBkY21hIFVzZXIiLCJEQVJRIGRpc2EgQ29vcmRpbmF0b3IgMSIsIlZpZXcgQWdlbmN5IEpTIiwiREFSUSBqb2ludHN0YWZmIEFkbWluIiwiREFSUSBtb2NrZGhwIEhRIiwiREFSUSBtb2NrdXNtYyBVc2VyIEFkbWluIiwiVmlldyBBZ2VuY3kgZm1zIiwiREFSUSBtb2NrZGFycGEgUmV2aWV3ZXIgNCIsIlZpZXcgQWdlbmN5IERPRElHIiwiREFSUSBtb2NrZGZhcyBSZXZpZXdlciAxIiwiREFSUSBtZGEgVXNlciIsIkRBUlEgZG1hIFJldmlld2VyIDIiLCJWaWV3IEFnZW5jeSBBaXIgTmF0aW9uYWwgR3VhcmQiLCJEQVJRIGRvZGlnIEFkbWluIiwiREFSUSBtb2NrbWRhIENvb3JkaW5hdG9yIDMiLCJWaWV3IEhERlMgUmVjb25zIiwiREFSUSBtb2NrbWRhIFRlc3RlciIsIkRBUlEgZGF1IFVzZXIiLCJWaWV3IERBUlEiLCJEQVJRIGRjbWEgVmlldyBPbmx5IiwiREFSUSBhZ2VuY3l4IFVzZXIgQWRtaW4iLCJEQVJRIG1vY2tuYXZ5IEFkbWluIiwiREFSUSBzb2NvbSBBZG1pbiIsIkRBUlEgT1NEIiwiREFSUSBkY21hIENvb3JkaW5hdG9yIDIiLCJWaWV3IEFnZW5jeSBESFJBIiwiVmlldyBBZ2VuY3kgVVNTT0NPTSIsIkRBUlEgYWdlbmN5eCBDb29yZGluYXRvciAyIiwiREFSUSBtb2NrZGFycGEgVmlldyBPbmx5IiwiVmlldyBBZ2VuY3kgV2hpdGUgSG91c2UiLCJEQVJRIG1vY2tqb2ludHN0YWZmIFZpZXcgT25seSIsIkRBUlEgbW9ja2FybXkgQ29vcmRpbmF0b3IgMSIsIkRBUlEgYXJteSBVc2VyIiwiVmlldyBBZ2VuY3kgREVDQSIsIlZpZXcgQWdlbmN5IEVPUE5TQyIsIkRBUlEgYXJteSBSZXZpZXdlciAyIiwiREFSUSBtb2NrbmF2eSBSZXZpZXdlciAzIiwiVmlldyBBZ2VuY3kgT3RoZXIgRGVmZW5zZSBBY3Rpdml0aWVzIFRpZXIgMiIsIkRBUlEgbW9ja21kYSBIUSIsImNhdGFsb2dNZXRhZGF0YVNlYXJjaFRyZW5kaW5nR0VUIiwiVmlldyBVc2VyIE1ldHJpY3MiLCJEQVJRIGRwYWEgQWRtaW4iLCJEQVJRIG1vY2tkaHAgVGVzdGVyIiwiVmlldyBBZ2VuY3kgVUNDIEhRIiwiREFSUSBtZGEgQWRtaW4iLCJEQVJRIGRhdSBWaWV3IE9ubHkiLCJEQVJRIGRmYXMgQ29vcmRpbmF0b3IgMiIsIlZpZXcgQWdlbmN5IERlcG90IE1haW50ZW5hbmNlIiwiREFSUSBtb2NrZGZhcyBVc2VyIiwiVmlldyBVc2VyIE1hbmFnZW1lbnQgUmVhZC1Pbmx5IiwiREFSUSBtb2NrZGFycGEgUmV2aWV3ZXIgMiIsIkVkaXQgRGF0YSBTdGF0dXMgVHJhY2tlciIsIkRBUlEgbW9ja2pvaW50c3RhZmYgSFEiLCJEQVJRIGRzcyBVc2VyIiwiREFSUSBuYXZ5IFJldmlld2VyIDIiLCJWaWV3IEFnZW5jeSBCVEEiLCJEQVJRIGR0cmEgR3JvdXAgQWRtaW4iLCJWaWV3IEFnZW5jeSBEVFNBIiwiVmlldyBBZ2VuY3kgbW9ja25hdnkiLCJWaWV3IEFnZW5jeSBBaXIgRm9yY2UgR2VuZXJhbCBGdW5kIiwiVmlldyBBZ2VuY3kgQ1BNUyIsIlZpZXcgVHJpZmFjdGEiLCJEQVJRIGRmYXMgR3JvdXAgQWRtaW4iLCJEQVJRIG5hdnkgUmV2aWV3ZXIgMSIsIkRBUlEgbW9ja2RzY2EgQWRtaW4iXSwic2FuZGJveElkIjoxLCJkaXNhYmxlZCI6ZmFsc2UsImNzcmYtdG9rZW4iOiI0ZWE1YzUwOGE2NTY2ZTc2MjQwNTQzZjhmZWIwNmZkNDU3Nzc3YmUzOTU0OWM0MDE2NDM2YWZkYTY1ZDIzMzBlIiwiaWF0IjoxNjQzOTIwOTc2fQ.joAkdgvZkfbzDiXM3KtX6ieIiqKWozEwnZZR-6FUj8Y1LLop4FPFMCYBktv3GD_aHR8rqUI0Sq22M8AgI_TRF-3BlSImRF2FzlU2e1sJzKT1wQT1MSJX0b7C4Tzq1m9K7C9MSi-cmD2W2VEgfnlRCQfSWYLEwKHMOrGv0HDedkHiag5pV9ezQjs0lyEmvD-W7ZWuJdS9WQJsdT2i4Q49at_nHkDKpHF_VgKVEBUZlmhRilZcLZVET2YFro21xx7utgCt_6uZ3cWDfOevrxfSnzdtR-gctpj5LBKIaTLvvQNHiQJaPTjhKG7VINCaQ0x2CKe5pd4cXdtZd9LHLJZOnZQza6hdtRTWiHyAvpUCGac_TfUY7jWC2-dtLXCAeLlRbKLHR_oFdppE9GzrNGEBy54RyEDER1V68FQ6G040pHYgkYUAMD6gICRPi_n562EDEvkqKqMhGX6MiIPp8WVNT10gGh_ASzdbYScZS08Lcp_Ur4t3kOb8dYU3yMM8gty6JptyYy3uvRfo_oQ6mnO5ZwTrgu3uudc7TcuCJnDGpC_lfsVAjlEBHJItNNf1Qg7b0iuApyXvfMo--Z4dlAUQrZkeKCV9ODndr9UOCIsvLEkFtEEuDr4yk7nrDaHfMhWiMH_U7E2tyNGYobKtYaHsvUgTrSUBGmI0h5ou1aUXQJw"
    endpoint = "/api/gameChanger/admin/getProcessStatus"
    url = f"http://host.docker.internal:8990" + endpoint
    m = hashlib.sha256()
    m.update((endpoint+token).encode('utf-8'))
    logger.info(m.hexdigest())
    r = requests.get(url, headers={
        "X-UA-SIGNATURE":m.hexdigest()
    })
    r.raise_for_status()
    logger.info(r.content)
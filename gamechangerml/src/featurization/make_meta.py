import pandas as pd
import logging
from datetime import date
from gamechangerml.api.utils.logger import logger
from gamechangerml.src.featurization.rank_features.generate_ft import generate_ft_doc

logger = logging.getLogger()

try:
    import wikipedia
except Exception as e:
    logger.warning(e)
    logger.warning("Wikipedia may not be installed")

def make_pop_docs(user_data, save_path):
    '''Makes popular_documents.csv'''
    logger.info("| --------- Making popular documents csv from user search history ----- |")
    try:
        data = user_data.document.value_counts().to_frame().reset_index()
        data.rename(columns={"document": "pop_score", "index": "doc"}, inplace=True)
        data.to_csv(save_path, index=False)
        logger.info(f" *** Saved popular documents to {save_path}")
    except Exception as e:
        logger.info("Error making popular documents csv")
        logger.info(e)
    return

def make_combined_entities(topics, orgs, save_path):
    '''Makes combined_entities.csv'''
        
    def lookup_wiki_summary(query):
        '''Get summaries for topics and orgs from Wikipedia'''
        try:
            logger.info(f"Looking up {query}")
            return wikipedia.summary(query).replace("\n", "")
        except Exception as e:
            logger.info(f"Could not retrieve description for {query}")
            logger.info(e)
            return ""

    logger.info("| --------- Making combined entities csv (orgs and topics) -------- |")
    try:
        ## clean up orgs dataframe
        if "Unnamed: 0" in orgs.columns:
            orgs.drop(columns=["Unnamed: 0"], inplace=True)
        orgs.rename(columns={"Agency_Name": "entity_name"}, inplace=True)
        orgs["entity_type"] = "org"
        ## clean up topics dataframe
        topics.rename(
            columns={"name": "entity_name", "type": "entity_type"}, inplace=True
        )    
        combined_ents = orgs.append(topics)
        combined_ents["information"] = combined_ents["entity_name"].apply(
            lambda x: lookup_wiki_summary(x)
        )
        combined_ents["information_source"] = "Wikipedia"
        combined_ents["information_retrieved"] = date.today().strftime(
            "%Y-%m-%d")
        combined_ents.to_csv(save_path, index=False)
        logger.info(f" *** Saved combined entities to {save_path}")
    except Exception as e:
        logger.info("Error making combined entities csv")
        logger.info(e)
    return

def make_corpus_meta(corpus_dir, days, prod_data):
    '''Generates corpus_meta.csv of ranking features'''
    logger.info("| ------------  Making corpus_meta.csv (rank features) ------------- |")
    try:
        generate_ft_doc(corpus_dir, days, prod_data)
    except Exception as e:
        logger.info("Could not generate corpus meta file")
        logger.info(e)
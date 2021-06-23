import argparse
import logging
import os
from datetime import datetime

from gamechangerml.src.search.query_expansion.build_ann_cli import (
    build_qe_model as bqe,
)
from gamechangerml.src.utilities import utils
from gamechangerml.configs.config import DefaultConfig, D2VConfig
# from gamechangerml.src.search.sent_transformer.model import SentenceEncoder
import pandas as pd
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

modelname = datetime.now().strftime("%Y%m%d")

SEARCH_MAPPINGS_FILE = "gamechangerml/data/SearchPdfMapping.csv"
TOPICS_FILE = "gamechangerml/data/topics_wiki.csv"
ORGS_FILE = "gamechangerml/data/agencies/agencies_in_corpus.csv"
OUT_PATH =  "gamechangerml/data"
class Pipeline():
    def generate_meta_data(self,):
        try:
            mappings = pd.read_csv(SEARCH_MAPPINGS_FILE)
        except Exception as e:
            print(e)
        mappings = self.process_mappings(mappings)
        mappings.to_csv(os.path.join(OUT_PATH,"popular_documents.csv"), index = False)
        try:
            topics = pd.read_csv(TOPICS_FILE)
            orgs = pd.read_csv(ORGS_FILE)
        except Exception as e:
            print(e)
        orgs.drop(columns=["Unnamed: 0"], inplace=True)
        topics.rename(columns={"name": "entity_name",
                      "type": "entity_type"}, inplace=True)
        orgs.rename(columns={"Agency_Name": "entity_name"}, inplace=True)
        orgs["entity_type"] = "org"
        combined_ents = orgs.append(topics)
        combined_ents.to_csv(os.path.join(OUT_PATH, "combined_entities.csv"), index=False)

    def process_mappings(self,data):
        data = data.document.value_counts().to_frame().reset_index()
        data.rename(columns ={'document':'count', 'index':'doc'},inplace=True)
        return data
    def run_qexp(
        self,
        model_id,
        save_remote=False,
        corpus_dir=DefaultConfig.DATA_DIR,
        model_dest=DefaultConfig.LOCAL_MODEL_DIR,
        exp_name=modelname,
        validate=False,
        sentenceTrans=False,
        gpu=False,
    ):
        model_dir = model_dest

        # get model name schema
        model_id = utils.create_model_schema(model_dir, model_id)

        # todo: revise try/catch logic so mlflow_id is not referenced before assignment
        mlflow_id = None

        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except RuntimeError:
            logger.warning("MLFLOW may not be installed")
        # start experiment
        try:
            # try to create experiment by exp name
            mlflow_id = mlflow.create_experiment(name=exp_name)
        except:
            try:
                # if it exists set id
                mlflow_id = mlflow.get_experiment_by_name(exp_name).experiment_id
            except:
                # if mlflow does not exist
                logger.warning("cannot get experiment from MLFlow")
        # attempt mlflow start
        try:
            with mlflow.start_run(experiment_id=mlflow_id):
                # build ANN indices
                index_dir = os.path.join(model_dest, model_id)
                bqe.main(
                    corpus_dir,
                    index_dir,
                    num_trees=125,
                    num_keywords=2,
                    ngram=(1, 2),
                    word_wt_file="word-freq-corpus-20201101.txt",
                    abbrv_file=None,
                )
                for param in D2VConfig.MODEL_ARGS:
                    mlflow.log_param(param, D2VConfig.MODEL_ARGS[param])
                mlflow.log_param("model_id", model_id)
                logger.info(
                    "-------------- Model Training Complete --------------")
                logger.info(
                    "-------------- Building Sentence Embeddings --------------")
                if save_remote:
                    utils.save_all_s3(model_dest, model_id)

                if validate:
                    logger.info(
                        "-------------- Running Assessment Model Script --------------"
                    )

                    logger.info(
                        "-------------- Assessment is not available--------------")
                    """
                    results = mau.assess_model(
                        model_name=model_id,
                        logger=logger,
                        s3_corpus="corpus_20200909",
                        model_dir="gamechangerml/models/",
                        verbose=True,
                    )
                    for metric in results:
                        if metric != "model_name":
                            mlflow.log_metric(key=metric, value=results[metric])
                    """
                    logger.info(
                        "-------------- Finished Assessment --------------")
                else:
                    logger.info("-------------- No Assessment Ran --------------")

            mlflow.end_run()
        except Exception:
            # try only models without mlflow
            logger.info("-------------- Training without MLFLOW --------------")
            index_dir = os.path.join(model_dest, model_id)
            bqe.main(
                corpus_dir,
                index_dir,
                num_trees=125,
                num_keywords=2,
                ngram=(1, 2),
                word_wt_file="word-freq-corpus-20201101.txt",
                abbrv_file=None,
            )
            if save_remote:
                utils.save_all_s3(model_dest, model_id)
        logger.info("-------------- Model Training Complete --------------")



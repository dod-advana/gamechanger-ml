import json
import os
import pandas as pd
from datetime import datetime
from gamechangerml import CORPUS_PATH
from gamechangerml.configs import SemanticSearchConfig

from gamechangerml.src.search.semantic_search import SemanticSearch
from gamechangerml.src.utilities import (
    create_directory_if_not_exists,
    open_txt,
    save_json,
    open_json,
)
from ..validation_data import UpdatedGCRetrieverData
from .retriever_evaluator import RetrieverEvaluator
from .utils import LOCAL_TRANSFORMERS_DIR, logger


class IndomainRetrieverEvaluator(RetrieverEvaluator):
    def __init__(
        self,
        encoder_model_name,
        data_level,
        index,
        create_index=True,
        data_path=None,
        encoder=None,
        retriever=None,
        transformer_path=LOCAL_TRANSFORMERS_DIR,
        use_gpu=False,
    ):

        super().__init__(transformer_path, encoder_model_name, use_gpu)

        self.model_path = os.path.join(transformer_path, encoder_model_name)
        self.data_path = data_path
        self.data_level = data_level
        logger.info(f"Using {str(self.data_path)} for validation data")
        if not index:  # if there is no index to evaluate, we need to make one
            logger.info(
                "No index provided for evaluating. Checking if test index exists."
            )
            self.index_path = os.path.join(
                transformer_path, encoder_model_name, "sent_index_TEST"
            )
            # make evaluations path
            self.eval_path = create_directory_if_not_exists(
                os.path.join(self.model_path, "evals_gc", data_level)
            )
            if (
                os.path.isdir(self.index_path)
                and len(os.listdir(self.index_path)) > 0
            ):
                logger.info("Found a test index for this model, using that.")
            else:
                logger.info("Did not find a test index - creating one.")
                if (
                    create_index
                ):  # make test index in the encoder model directory
                    # create directory for the test index
                    if not os.path.exists(self.index_path):
                        os.makedirs(self.index_path)
                    logger.info(
                        "Making new embeddings index at {}".format(
                            str(self.index_path)
                        )
                    )

                    # set up the encoder to make the index
                    if encoder:  # if encoder model is passed, use that
                        logger.info(
                            f"Using pre-init encoder to make the index"
                        )
                        self.encoder = encoder
                    else:  # otherwise init an encoder to make the index
                        logger.info(
                            f"Loading {encoder_model_name} to make the index"
                        )
                        self.encoder = SemanticSearch(
                            self.model_path,
                            self.index_path,
                            False,
                            logger,
                            self.use_gpu,
                            SemanticSearchConfig.DEFAULT_THRESHOLD_ARG,
                        )

                    # create the test corpus
                    include_ids = self.collect_docs_for_index()
                    if len(include_ids) > 0:
                        logger.info(
                            f"Collected {str(len(include_ids))} doc IDs to include in test index"
                        )
                        logger.info(f"{str(include_ids[:5])}")
                    else:
                        logger.warning(
                            "Function to retrieve doc IDs didn't work"
                        )
                        quit

                    # make a (test) index for evaluating the model
                    logger.info("Making the test index")
                    self.make_index(
                        encoder=self.encoder,
                        corpus_path=CORPUS_PATH,
                        files_to_use=include_ids,
                    )

                    ## save index metadata
                    metadata = {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "model_type": "sentence index",
                        "base_model_path": self.model_path,
                        "current_model_path": self.index_path,
                        "validation_data_dir": self.data_path,
                        "include_ids": include_ids,
                    }
                    save_json("metadata.json", self.index_path, metadata)
                    logger.info("Saved metadata to the index dir")

            index = self.index_path
        else:  # if a full index is passed, use that for evaluating
            self.index_path = os.path.join(
                os.path.dirname(transformer_path), index
            )

            # make evaluations path
            self.eval_path = create_directory_if_not_exists(
                os.path.join(self.index_path, "evals_gc", data_level)
            )

        if self.index_path:  # at this point, there should be an index path
            # collect all the doc ids in the index
            self.doc_ids = open_txt(
                os.path.join(self.index_path, "doc_ids.txt")
            )

            # if retriever exists, use that, otherwise make one
            if retriever:
                self.retriever = retriever
            else:
                self.retriever = SemanticSearch(
                    self.model_path,
                    self.index_path,
                    True,
                    logger,
                    self.use_gpu,
                    SemanticSearchConfig.DEFAULT_THRESHOLD_ARG,
                )

            # make the validation data
            logger.info("Collecting query/result pairs for testing")
            self.data = UpdatedGCRetrieverData(
                available_ids=self.doc_ids,
                level=self.data_level,
                data_path=self.data_path,
            )

            logger.info("Generating results")
            # generate the evaluation results
            self.results = self.eval(
                data=self.data,
                index=index,
                retriever=self.retriever,
                data_name=data_level,
                eval_path=self.eval_path,
                model_name=encoder_model_name,
            )

    def collect_docs_for_index(self):
        """Check if the model has an associated training data file with IDs to include in test index."""

        if os.path.isfile(os.path.join(self.model_path, "metadata.json")):
            logger.info(
                "This is a finetuned model: collecting training data IDs for index"
            )
            metadata = open_json("metadata.json", self.model_path)
            train_data_path = metadata["training_data_dir"]
            training_data = pd.read_csv(train_data_path)
            include_ids = [
                i.split(".pdf_")[0] for i in list(set(training_data["doc"]))
            ]
        else:
            logger.info(
                "This is a base model: collecting validation data IDs for index"
            )
            base_val_path = os.path.join(self.data_path, self.data_level)
            validation_data = open_json(
                "intelligent_search_data.json", base_val_path
            )
            validation_data = json.loads(validation_data)
            include_ids = [
                i.strip().lstrip()
                for i in validation_data["collection"].values()
            ]

        include_ids = [
            i + ".json" if i[-5:] != "json" else i for i in include_ids
        ]
        return include_ids

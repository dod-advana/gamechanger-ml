import torch
import urllib3
from datetime import date
from distutils.dir_util import copy_tree
from json import dump, load
from os import environ, makedirs, PathLike, listdir, mkdir
from os.path import join, isdir
from pandas import read_csv
from threading import current_thread
from typing import Union, Dict

from gamechangerml import MODEL_PATH, DATA_PATH
from gamechangerml.src.search.ranking.ltr import LTR
from gamechangerml.src.search.semantic_search import SemanticSearch
from gamechangerml.src.featurization.topic_modeling import Topics
from gamechangerml.src.paths import (
    SEARCH_PDF_MAPPING_FILE,
    POPULAR_DOCUMENTS_FILE,
    TOPICS_FILE,
    ORGS_FILE,
    COMBINED_ENTITIES_FILE,
    PROD_DATA_FILE,
    CORPUS_DIR,
    DEFAULT_SENT_INDEX,
    SENT_TRANSFORMER_TRAIN_DIR,
)
from gamechangerml.src.utilities import (
    create_tgz_from_dir,
    create_model_schema,
    get_current_datetime,
    configure_logger,
)
from gamechangerml.src.paths import S3_DATA_PATH, S3_MODELS_PATH
from gamechangerml.src.services import S3Service
from gamechangerml.src.model_testing.evaluation import (
    IndomainRetrieverEvaluator,
)
from gamechangerml.src.search.semantic_search.train import (
    SemanticSearchFinetuner,
)
from gamechangerml.scripts.run_evaluation import (
    eval_qa,
    eval_sent,
    eval_sim,
    eval_qe,
)
from gamechangerml.src.featurization.make_meta import (
    make_pop_docs,
    make_combined_entities,
    make_corpus_meta,
)
from gamechangerml.scripts.make_training_data import make_training_data
from gamechangerml.src.utilities.test_utils import (
    get_user,
    get_index_size,
)
from gamechangerml.src.utilities import (
    open_json,
    get_most_recently_changed_dir,
)
from gamechangerml.api.utils import processmanager, status_updater
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.src.search.query_expansion.build_ann_cli import (
    build_qe_model as bqe,
)
from gamechangerml.configs import (
    SemanticSearchConfig,
    SimilarityConfig,
    QexpConfig,
    D2VConfig,
    S3Config,
)


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
environ["CURL_CA_BUNDLE"] = ""
environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"

logger = configure_logger(
    name=__name__,
    msg_fmt="%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s",
)
model_path_dict = get_model_paths()

LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
SENT_INDEX = model_path_dict["sentence"]

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception as e:
    logger.warning(f"MLFLOW may not be installed. {e}")


class Pipeline:
    def __init__(self):
        self.ltr = LTR()
        # read in input data files
        self._load_meta_files()

    def _load_meta_files(self):
        try:
            self.search_history = read_csv(SEARCH_PDF_MAPPING_FILE)
            self.topics = read_csv(TOPICS_FILE)
            self.orgs = read_csv(ORGS_FILE)
        except Exception as e:
            logger.exception(f"Failed to load meta file(s). {e}")

    def create_metadata(
        self,
        meta_steps,
        testing_only: bool,
        corpus_dir: Union[str, PathLike] = CORPUS_DIR,
        index_path: Union[str, PathLike] = DEFAULT_SENT_INDEX,
        days: int = 80,
        prod_data_file=PROD_DATA_FILE,
        level: str = "silver",
        update_eval_data: bool = True,
        upload: bool = True,
        version: str = "v1",
    ) -> None:
        """
        create_metadata: combines datasets to create readable sets for ingest
        Args:
            corpus_dir [Union[str,PathLike]]: path to corpus JSONs
            meta_steps [List[str]]: list of metadata steps to execute (
                options: ["pop_docs", "combined_ents", "rank_features", "update_sent_data"])
            days [int]: days back to go for creating rank features (** rank_features)
            prod_data_file [Union[str,PathLike]]: path to prod data file (** rank_features)
            index_path [Union[str,PathLike]]: sent index path (** update_sent_data)
            n_matching [int]: number of matching paragraphs to retrieve (** update_sent_data)
            level [str]: level of tiered eval data to use (any, silver, gold) (** update_sent_data)
            update_eval_data [bool]: whether or not to update the eval data (** update_sent_data)
        Returns:
            None (saves files for each step)
        """
        logger.info(f"Meta steps: {str(meta_steps)}")

        if "pop_docs" in meta_steps:
            make_pop_docs(self.search_history, POPULAR_DOCUMENTS_FILE)

        if "combined_ents" in meta_steps:
            make_combined_entities(
                self.topics, self.orgs, COMBINED_ENTITIES_FILE
            )

        if "rank_features" in meta_steps:
            make_corpus_meta(corpus_dir, days, prod_data_file, upload)

        if "update_sent_data" in meta_steps:
            try:
                make_training_data(
                    index_path=index_path,
                    level=level,
                    update_eval_data=update_eval_data,
                    testing_only=testing_only,
                )
            except Exception as e:
                logger.warning(e, exc_info=True)

        if upload:
            s3_path = join(S3_DATA_PATH, version)
            model_name = get_current_datetime()
            local_path = DATA_PATH + model_name + ".tar.gz"
            create_tgz_from_dir(src_dir=DATA_PATH, dst_archive=local_path)
            self.upload(s3_path, local_path, "data", model_name)

    def finetune_sent(
        self,
        batch_size: int = 8,
        epochs: int = 3,
        warmup_steps: int = 100,
        testing_only: bool = False,
        remake_train_data: bool = True,
        model=None,
        version: str = "v1",
    ) -> Dict[str, str]:
        """finetune_sent: finetunes the sentence transformer - saves new model,
           a csv file of old/new cos sim scores, and a metadata file.
        Args:
            batch_size [int]: batch_size
            epochs [int]: epochs
            warmup_steps [int]: warmup steps
            testing_only [bool]: set True if only testing finetune functionality
        Returns:
            metadata: meta information on finetuning
        """

        try:
            if not model:
                model_load_path = join(
                    LOCAL_TRANSFORMERS_DIR, SemanticSearchConfig.BASE_MODEL
                )
            else:
                model_load_path = join(LOCAL_TRANSFORMERS_DIR, model)

            logger.info(f"Model load path set to: {str(model_load_path)}")
            no_data = False
            # check if training data exists
            if remake_train_data:
                no_data = True
            # if no training data directory exists
            elif not isdir(SENT_TRANSFORMER_TRAIN_DIR):
                no_data = True
                makedirs(SENT_TRANSFORMER_TRAIN_DIR)
            elif (
                len(listdir(SENT_TRANSFORMER_TRAIN_DIR)) == 0
            ):  # if base dir exists but there are no files
                no_data = True
            elif (
                get_most_recently_changed_dir(SENT_TRANSFORMER_TRAIN_DIR)
                == None
            ):
                no_data = True
            elif (
                len(
                    listdir(
                        get_most_recently_changed_dir(
                            SENT_TRANSFORMER_TRAIN_DIR
                        )
                    )
                )
                == 0
            ):
                no_data = True
            logger.info(f"No data flag is set to: {str(no_data)}")

            # if we don't have data, make training data
            if no_data:
                make_training_data(
                    index_path=SENT_INDEX,
                    level="silver",
                    update_eval_data=True,
                    testing_only=testing_only,
                )

            data_path = get_most_recently_changed_dir(
                SENT_TRANSFORMER_TRAIN_DIR
            )
            timestamp = str(data_path).split("/")[-1]

            # set model save path
            if testing_only:
                model_save_path = model_load_path + "_TEST_" + timestamp
            else:
                model_id = get_current_datetime()
                model_save_path = model_load_path + "_" + model_id
            logger.info(
                f"Setting {str(model_save_path)} as save path for new model"
            )
            logger.info(f"Loading in domain data to finetune from {data_path}")
            finetuner = SemanticSearchFinetuner(
                model_load_path=model_load_path,
                model_save_path=model_save_path,
                data_directory=data_path,
                logger=logger,
                testing_only=testing_only,
                **SemanticSearchConfig.FINETUNE
            )
            logger.info("Loaded SemanticSearchFinetuner class...")
            logger.info(f"Testing only is set to: {testing_only}")

            # finetune
            finetuner.train(version)

            # eval finetuned model
            logger.info("Done making finetuned model, runnin evals")
            model_name = model_save_path.split("/")[-1]
            train_meta = open_json("training_metadata.json", data_path)
            validation_data = train_meta["validation_data_used"].split("/")[-1]
            evals = eval_sent(model_name, validation_data, eval_type="domain")

            try:
                for metric in evals:
                    if metric != "model_name":
                        mlflow.log_metric(key=metric, value=evals[metric])
            except Exception as e:
                logger.warning(e)

            return evals

        except Exception as e:
            logger.warning("Could not finetune sentence model - pipeline")
            logger.error(e)

            raise e

    def evaluate(
        self,
        model_name: str,
        sample_limit: int,
        validation_data: str = "latest",
        eval_type: str = "domain",
        retriever=None,
        upload: bool = True,
        version: str = "v1",
    ) -> Dict[str, str]:
        """Evaluates models/sent index
        Args:
            model_name [str]: name of the model or sent index to evaluate
            sample_limit [int]: max samples for evaluating (if testing on original data)
            validation_data [str]: which validation set to use (
                "latest" pulls newest, otherwise needs timestamp dir ex. '2021-12-01_191740')
            eval_type [str]: type of evaluation to run (options = ["original", "domain"])
        Returns:
            eval [Dict[str,str]]: evaluation dictionary
        """
        results = {"original": {}, "domain": {}}
        try:
            logger.info(f"Attempting to evaluate model {model_name}")

            if "bert-base-cased-squad2" in model_name:
                results[eval_type] = eval_qa(
                    model_name, sample_limit, eval_type
                )
            elif "msmarco-distilbert" in model_name:
                for e_type in ["domain", "original"]:
                    results[e_type] = eval_sent(
                        model_name, validation_data, e_type, retriever
                    )
            elif "multi-qa-MiniLM" in model_name:
                results["domain"] = eval_sent(
                    model_name,
                    validation_data,
                    eval_type="domain",
                    retriever=retriever,
                )
            elif "sent_index" in model_name:
                results["domain"] = eval_sent(
                    model_name,
                    validation_data,
                    eval_type="domain",
                    retriever=retriever,
                )
            elif "distilbart-mnli-12-3" in model_name:
                results[eval_type] = eval_sim(
                    model_name, sample_limit, eval_type
                )
            elif "qexp" in model_name:
                results["domain"] = eval_qe(model_name)
            else:
                logger.warning(
                    "There is currently no evaluation pipeline for this type of model."
                )
                raise Exception("No evaluation pipeline available")

        except Exception as e:
            logger.warning(f"Could not evaluate {model_name}")
            logger.warning(e)

        return results

    def create_qexp(
        self,
        model_id=None,
        upload=False,
        corpus=CORPUS_DIR,
        model_dest=MODEL_PATH,
        validate=True,
        version="v4",
    ):
        """
        create_qexp: creates a query expansion model
        Args:
            params for qexp configuration
        Returns:
            metadata: params or meta information for qexp
            evals: evaluation results dict
        """
        model_dir = model_dest

        if not model_id:
            model_id = get_current_datetime()

        # get model name schema
        model_name = "qexp_" + model_id
        model_path = create_model_schema(model_dir, model_name)
        evals = {"results": ""}
        params = D2VConfig.MODEL_ARGS
        try:
            # build ANN indices
            index_dir = join(model_dest, model_path)
            bqe.main(corpus, index_dir, **QexpConfig.BUILD_ARGS)
            logger.info(
                "-------------- Model Training Complete --------------"
            )
            # Create .tgz file
            dst_path = index_dir + ".tar.gz"
            create_tgz_from_dir(src_dir=index_dir, dst_archive=dst_path)

            logger.info(f"Created tgz file and saved to {dst_path}")

            if upload:
                s3_path = join(S3_MODELS_PATH, f"qexp_model/{version}")
                self.upload(s3_path, dst_path, "qexp", model_id)

            if validate:
                logger.info(
                    "-------------- Running Assessment Model Script --------------"
                )
                # qxpeval = QexpEvaluator(qe_model_dir=index_dir, **QexpConfig.INIT_ARGS, **QexpConfig.EXPANSION_ARGS, model=None)
                # evals = qxpeval.results

                logger.info(
                    "-------------- Assessment is not available--------------"
                )
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
                        mlflow.log_metric(
                            key=metric, value=results[metric])
                """

                logger.info(
                    "-------------- Finished Assessment --------------"
                )
            else:
                logger.info("-------------- No Assessment Ran --------------")
        except Exception as e:
            logger.error(e)
            logger.error("Error with QExp building")
        return params, evals

    def create_embedding(
        self,
        corpus,
        existing_embeds=None,
        encoder_model="msmarco-distilbert-base-v2_2021-10-17",
        gpu=True,
        upload=False,
        version="v4",
        validate=True,
    ):
        """
        create_embedding: creates a sentence embedding
        Args:
            params for sentence configuration
        Returns:
            metadata: params or meta information for qexp
            evals: evaluation results dict
        """
        # Error fix for saving index and model to tgz
        # https://github.com/huggingface/transformers/issues/5486
        try:
            environ["TOKENIZERS_PARALLELISM"] = "false"
        except Exception as e:
            logger.warning(e)
        logger.info("Entered create embedding")

        # GPU check
        use_gpu = gpu
        if use_gpu and not torch.cuda.is_available:
            logger.info(
                "GPU is not available. Setting `gpu` argument to False"
            )
            use_gpu = False

        # Define model saving directories
        model_id = get_current_datetime()
        model_name = "sent_index_" + model_id
        local_sent_index_dir = join(MODEL_PATH, model_name)

        # Define new index directory
        if not isdir(local_sent_index_dir):
            mkdir(local_sent_index_dir)
        logger.info(
            "-------------- Building Sentence Embeddings --------------"
        )
        logger.info("Loading Encoder Model...")

        # If existing index exists, copy content from reference index
        if existing_embeds is not None:
            copy_tree(existing_embeds, local_sent_index_dir)

        # Building the Index
        try:

            encoder = SemanticSearch(
                join(LOCAL_TRANSFORMERS_DIR, encoder_model),
                local_sent_index_dir,
                False,
                logger,
                use_gpu,
                None
            )
            logger.info(
                f"Creating Document Embeddings with {encoder_model} on {corpus}"
            )
            logger.info("-------------- Indexing Documents--------------")
            start_time = get_current_datetime("%Y-%m-%d %H:%M:%S")
            encoder_corpus = encoder.prepare_corpus_for_embedding(corpus)
            encoder.create_embeddings_index(encoder_corpus)

            end_time = get_current_datetime("%Y-%m-%d %H:%M:%S")
            logger.info("-------------- Completed Indexing --------------")
            user = get_user(logger)

            # Checking length of IDs
            try:
                SENT_INDEX_PATH = model_path_dict["sentence"]
                old_index_len = get_index_size(SENT_INDEX_PATH)
                new_index_len = len(encoder.embedder.config["ids"])
                if new_index_len < old_index_len:
                    logger.warning(
                        f"Length of index ({str(new_index_len)}) is shorter than previous index ({str(old_index_len)})"
                    )
                    logger.info(f"Old index location: {str(SENT_INDEX_PATH)}")
            except Exception as e:
                logger.warning(
                    f"Could not compare length to old index: {str(SENT_INDEX_PATH)}"
                )
                logger.error(e)

            # Generating process metadata
            metadata = {
                "user": user,
                "date_started": start_time,
                "date_finished": end_time,
                "doc_id_count": len(encoder.embedder.config["ids"]),
                "corpus_name": corpus,
                "encoder_model": encoder_model,
            }

            # Create metadata file
            metadata_path = join(local_sent_index_dir, "metadata.json")
            with open(metadata_path, "w") as fp:
                dump(metadata, fp)

            logger.info(f"Saved metadata.json to {metadata_path}")

            logger.info("-------------- Running Evaluation --------------")
            try:
                evals = {}
                for level in ["gold", "silver"]:
                    sentev = IndomainRetrieverEvaluator(
                        encoder=encoder,
                        index=model_name,
                        data_level=level,
                        encoder_model_name=EmbedderConfig.BASE_MODEL,
                        sim_model_name=SimilarityConfig.BASE_MODEL,
                        **EmbedderConfig.MODEL_ARGS,
                    )
                    evals[level] = sentev.results
                    logger.info(
                        f"Evals for {level} standard validation: {(str(sentev.results))}"
                    )
            except Exception as e:
                logger.warning(
                    "Could not create evaluations for the new sentence index"
                )
                logger.error(e)

            # Create .tgz file
            dst_path = local_sent_index_dir + ".tar.gz"
            create_tgz_from_dir(
                src_dir=local_sent_index_dir, dst_archive=dst_path
            )

            logger.info(f"Created tgz file and saved to {dst_path}")
            logger.info(
                "-------------- Finished Sentence Embedding--------------"
            )
        except Exception as e:
            logger.warning("Error with creating embedding")
            logger.error(e)
        # Upload to S3
        if upload:
            s3_path = join(
                S3_MODELS_PATH,
                f"sentence_index/{version}",
                f"sentence_index_{model_id}.tar.gz",
            )
            bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)
            S3Service.upload_file(bucket, dst_path, s3_path, logger)
        return metadata, evals

    def init_ltr(self):
        try:
            ltr = self.ltr
            logger.info("attempting to init LTR")
            resp = ltr.post_init_ltr()
            logger.info(resp)
            logger.info("attempting to post features to ES")
            resp = ltr.post_features()
            logger.info(resp)
        except Exception as e:
            logger.warning(e)
            logger.warning("Could not initialize LTR")

    def create_ltr(self, daysBack: int = 180):
        try:
            ltr = self.ltr
            processmanager.update_status(
                processmanager.ltr_creation,
                0,
                4,
                thread_id=current_thread().ident,
            )
            logger.info("Attempting to create judgement list")
            # NOTE: always set it false right now since there needs to be API changes in the WEB
            remote_mappings = False
            # if environ.get("ENV_TYPE") == "PROD":
            #    remote_mappings = True
            judgements = ltr.generate_judgement(
                remote_mappings=remote_mappings, daysBack=daysBack
            )
            processmanager.update_status(
                processmanager.ltr_creation,
                1,
                4,
                thread_id=current_thread().ident,
            )
            logger.info("Attempting to get features")
            fts = ltr.generate_ft_txt_file(judgements)
            processmanager.update_status(
                processmanager.ltr_creation,
                2,
                4,
                thread_id=current_thread().ident,
            )
            logger.info("Attempting to read in data")
            ltr.data = ltr.read_xg_data()
            logger.info("Attempting to train LTR model")
            bst, model = ltr.train()
            processmanager.update_status(
                processmanager.ltr_creation,
                3,
                4,
                thread_id=current_thread().ident,
            )
            logger.info("Created LTR model")
            with open(join(MODEL_PATH, "ltr/xgb-model.json")) as f:
                model = load(f)
            logger.info("removing old LTR")
            resp = ltr.delete_ltr("ltr_model")
            logger.info(resp)
            resp = ltr.post_model(model, model_name="ltr_model")
            logger.info("Posted LTR model")
            processmanager.update_status(
                processmanager.ltr_creation,
                4,
                4,
                thread_id=current_thread().ident,
            )
        except Exception as e:
            logger.error("Could not create LTR")

    def create_topics(
        self,
        sample_rate=None,
        upload=False,
        corpus_dir=CORPUS_DIR,
        version="v2",
    ):
        try:
            model_id = get_current_datetime("%Y%m%d%H%M%S")

            # get model name schema
            model_name = "topic_model_" + model_id

            local_dir = join(MODEL_PATH, model_name)
            # Define new index directory
            if not isdir(local_dir):
                mkdir(local_dir)

            # Train topics
            status = status_updater.StatusUpdater(
                process_key=processmanager.topics_creation,
                nsteps=6,
            )
            topics_model = Topics(status=status)
            metadata = topics_model.train_from_files(
                corpus_dir=corpus_dir,
                sample_rate=sample_rate,
                local_dir=local_dir,
            )

            # Create metadata file
            metadata_path = join(local_dir, "metadata.json")
            with open(metadata_path, "w") as fp:
                dump(metadata, fp)

            # Create .tar.gz file from dir
            tar_path = local_dir + ".tar.gz"
            create_tgz_from_dir(src_dir=local_dir, dst_archive=tar_path)

            logger.info(f"create_topics complete, should upload? {upload}")
            # Upload to S3
            if upload:
                s3_path = join(S3_MODELS_PATH, f"topic_model/{version}")
                logger.info(f"Topics uploading to {s3_path}")
                self.upload(s3_path, tar_path, "topic_model", model_id)

            evals = None  # TODO: figure out how to evaluate this
            return metadata, evals

        except Exception as e:
            logger.error(f"Could not create topics {e}")

    def upload(self, s3_path, local_path, model_prefix, model_name):
        # Loop through each file and upload to S3
        logger.info(f"Uploading files to {s3_path}\n\tUploading: {local_path}")
        s3_path = join(s3_path, f"{model_prefix}_" + model_name + ".tar.gz")
        logger.info(f"s3_path {s3_path}")
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)
        S3Service.upload_file(
            bucket=bucket,
            filepath=local_path,
            s3_fullpath=s3_path,
            logger=logger,
        )
        logger.info(f"Successfully uploaded files to {s3_path}")
        logger.info("-------------- Finished Uploading --------------")

    def run(self, build_type, run_name, params):
        """
        run: record results of params and evaulations
        Args:
        Returns:
        """
        try:
            mlflow.create_experiment(str(date.today()))

        except Exception as e:
            logger.warning(e)
            logger.warning("Could not create experiment")
        metadata = evals = {}
        try:
            with mlflow.start_run(run_name=run_name) as run:
                if build_type == "sent_finetune":
                    metadata = self.finetune_sent(**params)
                elif build_type == "sentence":
                    metadata, evals = self.create_embedding(**params)
                elif build_type == "qexp":
                    metadata, evals = self.create_qexp(**params)
                elif build_type == "eval":
                    evals = self.evaluate(**params)
                elif build_type == "topics":
                    metadata, evals = self.create_topics(**params)
                elif build_type == "meta":
                    self.create_metadata(**params)

                self.mlflow_record(metadata, evals)
                processmanager.update_status(
                    processmanager.training,
                    0,
                    1,
                    f"training {build_type} model",
                    thread_id=current_thread().ident,
                )

            mlflow.end_run()
            processmanager.update_status(
                processmanager.training,
                1,
                1,
                f"trained {build_type} model",
                thread_id=current_thread().ident,
            )
        except Exception as e:
            logger.warning(f"Error building {build_type} with MLFlow")
            logger.warning(e)
            logger.warning(f"Trying without MLFlow")
            try:
                if build_type == "sent_finetune":
                    metadata = self.finetune_sent(**params)
                elif build_type == "sentence":
                    metadata, evals = self.create_embedding(**params)
                elif build_type == "qexp":
                    metadata, evals = self.create_qexp(**params)
                elif build_type == "eval":
                    evals = self.evaluate(**params)
                elif build_type == "topics":
                    metadata, evals = self.create_topics(**params)
                elif build_type == "meta":
                    self.create_metadata(**params)
                else:
                    logger.info(
                        f"Started pipeline with unknown build_type: {build_type}"
                    )
                processmanager.update_status(
                    processmanager.training,
                    0,
                    1,
                    f"training {build_type} model",
                    thread_id=current_thread().ident,
                )
                processmanager.update_status(
                    processmanager.training,
                    1,
                    1,
                    f"trained {build_type} model",
                    thread_id=current_thread().ident,
                )
            except Exception as err:
                logger.error("Could not train %s" % build_type)
                logger.error(err)
                processmanager.update_status(
                    processmanager.loading_corpus,
                    message="failed to load corpus",
                    failed=True,
                    thread_id=current_thread().ident,
                )
                processmanager.update_status(
                    processmanager.training,
                    message="failed to train " + build_type + " model",
                    failed=True,
                    thread_id=current_thread().ident,
                )

    def mlflow_record(self, metadata, evals):
        """
        mlflow_record: record results of params and evaulations
        Args:
        Returns:
        """
        for param in metadata:
            mlflow.log_param(param, metadata[param])
        for metric in evals:
            try:
                mlflow.log_metric(metric, evals[metric])
            except Exception as e:
                logger.warning(f"could not log metric: {metric}")

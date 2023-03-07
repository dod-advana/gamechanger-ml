from os.path import join
from json import dump
from logging import getLogger
import torch
from torch.utils.data import DataLoader
from threading import current_thread
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CosineSimilarityLoss
from datetime import datetime
from gamechangerml.src.paths import S3_MODELS_PATH, S3_DATA_PATH
from gamechangerml.src.services import S3Service
from gamechangerml.configs import S3Config
from gamechangerml.src.utilities import (
    open_json,
    save_json,
    create_tgz_from_dir,
)
from gamechangerml.api.utils import processmanager
from .semantic_search_training_data import SemanticSearchTrainingData

torch.cuda.empty_cache()


class SemanticSearchFinetuner:
    def __init__(
        self,
        model_load_path,
        model_save_path,
        data_directory,
        shuffle,
        batch_size,
        epochs,
        warmup_steps,
        logger=None,
        testing_only=False,
    ):
        """Finetune the sentence transformer Semantic Search model.

        Args:
            model_load_path (str): Path to load the model from.
            model_save_path (str): Path to save the finetuned model to.
            data_directory (str): Path to directory with training data.
            shuffle (bool): True for the data to be reshuffled at every epoch.
            batch_size (int): How many samples per batch to load.
            epochs (int): Number of epochs for training.
            warmup_steps (int): The learning rate is increased from 0 up to the
                maximal learning rate. After these many training steps, the
                learning rate is decreased linearly back to 0.
            logger (logging.Logger or None, optional): Defaults to None.
            testing_only (bool, optional): True to test out this class with
                30 train items and 10 test items and no upload to S3. Defaults
                to False.
        """
        self.model_load_path = model_load_path
        self.model = SentenceTransformer(model_load_path)
        self.model_save_path = model_save_path
        self.data_directory = data_directory
        self.testing_only = testing_only

        if not logger:
            logger = getLogger(__name__)
        self.logger = logger

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps

        self.fix_model_config_for_train_version(model_load_path, logger)

    @staticmethod
    def fix_model_config_for_train_version(model_path, logger):
        """Workaround for error with sentence_transformers==0.4.1
        (vs. version 2.0.0 which our model was trained on).
        """
        config_path = join(model_path, "config.json")
        alternate_config_path = join(
            model_path, "config_sentence_transformers.json"
        )

        try:
            config = open_json(config_path)
        except:
            logger.exception(f"Failed to load config file: {config_path}")
            return

        key = "__version__"
        if key in config:
            return

        try:
            alternate_config = open_json(alternate_config_path)
            config[key] = alternate_config[key]["sentence_transformers"]
        except:
            config[key] = "2.0.0"

        with open(config_path, "w") as f:
            dump(config, f)

    def train(self, version):
        try:
            self.logger.info("Starting training")
            self._update_process_manager_status(False)

            self.training_data = SemanticSearchTrainingData(
                self.data_directory, self.logger, self.testing_only
            )

            self.logger.info("Starting DataLoader")
            data_loader = DataLoader(
                self.training_data.samples,
                shuffle=self.shuffle,
                batch_size=self.batch_size,
            )
            loss = CosineSimilarityLoss(model=self.model)

            self.logger.info("Finetuning the encoder model")
            self.model.fit(
                train_objectives=[(data_loader, loss)],
                epochs=self.epochs,
                warmup_steps=self.warmup_steps,
            )

            self._update_process_manager_status(True)

            self.model.save(self.model_save_path)
            self.logger.info(
                f"Saved finetuned model to: `{self.model_save_path}`"
            )

            self._timestamp = datetime.now().strftime("%Y%m%d")

            self._save_metadata(version)

            if not self.testing_only:
                self._save_to_s3()
        except Exception:
            self.logger.exception("Failed to complete finetuning.")

    def _save_metadata(self, version):
        metadata = {
            "date": self._timestamp,
            "model_type": "finetuned encoder",
            "base_model_path": self.model_load_path,
            "current_model_path": self.model_save_path,
            "training_data_dir": self.training_data.csv_path,
            "n_training_samples": self.training_data.total,
            "version": version,
            "testing_only": self.testing_only,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
        }
        save_json("metadata.json", self.model_save_path, metadata)
        self.logger.info(
            f"Saved finetune metadata to: `{self.model_save_path}/metadata.json`"
        )

    def _save_to_s3(self, version):
        s3_bucket = S3Service.connect_to_bucket(
            S3Config.BUCKET_NAME, self.logger
        )

        # Create data tar & upload it to S3.
        data_path = self.data_directory + ".tar.gz"
        create_tgz_from_dir(self.data_directory, data_path)
        s3_data_path = join(
            S3_DATA_PATH, version, "data_", self._timestamp, ".tar.gz"
        )
        self.logger.info(f"Uploading data to S3: {s3_data_path}")
        S3Service.upload_file(s3_bucket, data_path, s3_data_path, self.logger)

        # Create model tar & upload it to S3.
        model_path = self.model_save_path + ".tar.gz"
        create_tgz_from_dir(self.model_save_path, model_path)
        model_id = self.model_save_path.split("_")[1]
        s3_model_path = join(
            S3_MODELS_PATH, str(version, f"transformers_{model_id}.tar.gz")
        )
        self.logger.info(f"Uploading model to S3: {s3_model_path}.")
        S3Service.upload_file(
            s3_bucket, model_path, s3_model_path, self.logger
        )

    def _update_process_manager_status(self, finished: bool):
        processmanager.update_status(
            processmanager.training,
            int(finished),
            1,
            thread_id=current_thread().ident,
        )

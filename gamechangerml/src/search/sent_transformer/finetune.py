from sentence_transformers import SentenceTransformer, InputExample, util, losses
from torch.utils.data import DataLoader
import pandas as pd
from datetime import date
import os
import json

S3_DATA_PATH = "bronze/gamechanger/ml-data"

from gamechangerml.src.utilities.test_utils import open_json, timestamp_filename, cos_sim
from gamechangerml.src.utilities import utils as utils
from gamechangerml.api.utils import processmanager
from gamechangerml.api.utils.logger import logger
from gamechangerml import DATA_PATH

def fix_model_config(model_load_path):
    """Workaround for error with sentence_transformers==0.4.1 (vs. version 2.0.0 which our model was trained on)"""

    try:
        config = open_json("config.json", model_load_path)
        if "__version__" not in config.keys():
            try:
                st_config = open_json("config_sentence_transformers.json", model_load_path)
                version = st_config["__version__"]["sentence_transformers"]
                config["__version__"] = version
            except:
                config["__version__"] = "2.0.0"
            with open(os.path.join(model_load_path, "config.json"), "w") as outfile:
                json.dump(config, outfile)
    except:
        logger.info("Could not update model config file")

def get_cos_sim(model, pair):

    emb1 = model.encode(pair[0])
    emb2 = model.encode(pair[1])
    try:
        sim = float(util.cos_sim(emb1, emb2))
    except:
        sim = float(cos_sim(emb1, emb2))
    
    return sim

def format_inputs(train, test):
    """Create input data for dataloader and df for tracking cosine sim"""

    train_samples = []
    all_data = []
    count = 0
    for i in train.keys():
        texts = [train[i]["query"], train[i]["paragraph"]]
        score = float(train[i]["label"])
        inputex = InputExample(str(count), texts, score)
        train_samples.append(inputex)
        all_data.append([i, texts, score, "train"])
        count += 1
    
    for x in test.keys():
        texts = [test[x]["query"], test[x]["paragraph"]]
        score = float(test[x]["label"])
        all_data.append([x, texts, score, "test"])

    df = pd.DataFrame(all_data, columns = ["key", "pair", "score", "label"])
    
    return train_samples, df

class STFinetuner():

    def __init__(self, model_load_path, model_save_path, shuffle, batch_size, epochs, warmup_steps):

        fix_model_config(model_load_path)
        self.model = SentenceTransformer(model_load_path)
        self.model_save_path = model_save_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
    
    def retrain(self, data_dir, testing_only, version):

        try:
            data = open_json("training_data.json", data_dir)
            train = data["train"]
            test = data["test"]

            if testing_only:
                logger.info("Creating smaller dataset just for testing finetuning.")
                train_keys = list(train.keys())[:3]
                test_keys = list(test.keys())[:3]
                train = {k:train[k] for k in train_keys}
                test = {k:test[k] for k in test_keys}

            ## make formatted training data
            train_samples, df = format_inputs(train, test)
            
            ## get cosine sim before finetuning
            df["original_cos_sim"] = df["pair"].apply(lambda x: get_cos_sim(self.model, x))

            ## finetune on samples
            processmanager.update_status(processmanager.loading_data, 0, 1)
            logger.info("Loading data for finetuning")
            train_dataloader = DataLoader(train_samples, shuffle=self.shuffle, batch_size=self.batch_size)
            train_loss = losses.CosineSimilarityLoss(model=self.model)
            processmanager.update_status(processmanager.loading_data, 1, 0)
            processmanager.update_status(processmanager.training, 0, 1) 
            logger.info("Finetuning the encoder model")
            self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.epochs, warmup_steps=self.warmup_steps)
            processmanager.update_status(processmanager.training, 1, 0)
            ## save model
            self.model.save(self.model_save_path)
            logger.info("Finetuned model saved to {}".format(str(self.model_save_path)))

            # when not testing only, save to S3
            if not testing_only:
                dst_path = self.model_save_path + ".tar.gz"
                utils.create_tgz_from_dir(src_dir=self.model_save_path, dst_archive=dst_path)
                model_id = self.model_save_path.split('_')[1]
                logger.info(f"Created tgz file and saved to {dst_path}")

                S3_MODELS_PATH = "bronze/gamechanger/models"
                s3_path = os.path.join(S3_MODELS_PATH, str(version))
                utils.upload(s3_path, dst_path, "transformers", model_id, version)
                logger.info(f"Saved model to S3: {s3_path}")

            ## get new cosine sim
            df["new_cos_sim"] = df["pair"].apply(lambda x: get_cos_sim(self.model, x))
            df["change_cos_sim"] = df["new_cos_sim"] - df["original_cos_sim"]

            ## save all results to CSV
            df.to_csv(os.path.join(data_dir, timestamp_filename("finetuning_results", ".csv")))

            ## create training metadata
            positive_change_train = df[(df["score"]==1.0) & (df["label"]=="train")]["change_cos_sim"].median()
            negative_change_train = df[(df["score"]==-1.0) & (df["label"]=="train")]["change_cos_sim"].median()
            neutral_change_train = df[(df["score"]==0.0) & (df["label"]=="train")]["change_cos_sim"].median() 
            positive_change_test = df[(df["score"]==1.0) & (df["label"]=="test")]["change_cos_sim"].median()
            negative_change_test = df[(df["score"]==-1.0) & (df["label"]=="test")]["change_cos_sim"].median()
            neutral_change_test = df[(df["score"]==0.0) & (df["label"]=="test")]["change_cos_sim"].median() 

            ft_metadata = {
                "date_finetuned": str(date.today()),
                "data_dir": str(data_dir),
                "positive_change_train": positive_change_train,
                "negative_change_train": negative_change_train,
                "neutral_change_train": neutral_change_train,
                "positive_change_test": positive_change_test,
                "negative_change_test": negative_change_test,
                "neutral_change_test": neutral_change_test
            }

            ## save metadata file
            ft_metadata_path = os.path.join(data_dir, timestamp_filename("finetuning_metadata", ".json"))
            with open(ft_metadata_path, "w") as outfile:
                json.dump(ft_metadata, outfile)

            logger.info("Metadata saved to {}".format(ft_metadata_path))
            logger.info(str(ft_metadata))

            # when not testing only, save to S3
            if not testing_only:
                s3_path = os.path.join(S3_DATA_PATH, f"{version}")
                logger.info(f"****    Saving new data files to S3: {s3_path}")
                dst_path = DATA_PATH + ".tar.gz"
                utils.create_tgz_from_dir(src_dir=DATA_PATH, dst_archive=dst_path)
                utils.upload_data(s3_path, dst_path)

            return ft_metadata
        
        except Exception as e:
            logger.warning("Could not finetune - finetune.py")
            logger.error(e, exc_info=True)
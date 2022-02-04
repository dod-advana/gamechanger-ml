from sentence_transformers import SentenceTransformer, InputExample, util, losses
from torch.utils.data import DataLoader
import pandas as pd
from datetime import date
import sys
import os
import json
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
torch.cuda.empty_cache()
import gc
import logging
import tqdm
from time import sleep
from gamechangerml.src.utilities.test_utils import open_json, timestamp_filename, cos_sim
from gamechangerml.src.utilities import utils as utils
from gamechangerml.src.model_testing.metrics import reciprocal_rank_score, get_MRR
from gamechangerml.api.utils.logger import logger
from datetime import datetime
from gamechangerml.api.utils import processmanager
from gamechangerml import DATA_PATH

S3_DATA_PATH = "bronze/gamechanger/ml-data"

logging.root.addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(force=True)
logger.setLevel(logging.INFO)

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
    total = len(train.keys()) + len(test.keys())
    for i in train.keys():
        texts = [train[i]["query"], train[i]["paragraph"]]
        score = float(train[i]["label"])
        inputex = InputExample(str(count), texts, score)
        train_samples.append(inputex)
        all_data.append([train[i]["query"], i, texts, score, "train"])
        count += 1
        processmanager.update_status(processmanager.loading_data, count, total)
    
    for x in test.keys():
        texts = [test[x]["query"], test[x]["paragraph"]]
        score = float(test[x]["label"])
        all_data.append([test[x]["query"], x, texts, score, "test"])
        count += 1
        processmanager.update_status(processmanager.loading_data, count, total)

    df = pd.DataFrame(all_data, columns = ["query", "key", "pair", "score", "label"])
    
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def summarize_results(self, df, data_dir):

        df["new_cos_sim"] = df["pair"].apply(lambda x: get_cos_sim(self.model, x))
        df["change_cos_sim"] = df["new_cos_sim"] - df["original_cos_sim"]
        df['score'] = df['score'].apply(lambda x: int(x))

        ## save all results to CSV
        df.to_csv(os.path.join(data_dir, timestamp_filename("finetuning_results", ".csv")))

        queries = list(set(df['query']))
        train_queries = list(set(df[df['label']=='train']['query'].tolist()))
        test_queries = [i for i in queries if i not in train_queries]
        
        def get_stats(df, query):
        
            mydict = {}
            sub = df[df['query']==query].copy()
            balance = sub['score'].value_counts().to_dict()
            ## calculate original scores
            sub = sub.sort_values(by = 'original_cos_sim', ascending = False)
            original_RR = reciprocal_rank_score(list(sub['score']))

            ## calculate new scores
            sub = sub.sort_values(by = 'new_cos_sim', ascending = False)
            new_RR = reciprocal_rank_score(list(sub['score']))

            mydict["query"] = query
            mydict["balance"] = balance
            mydict["original_RR"] = original_RR
            mydict["new_RR"] = new_RR

            return mydict
        
        results_dict = {}
        for q in queries:
            results = get_stats(df, q)
            results_dict[q] = results 
        
        summary = pd.DataFrame(results_dict)
        
        summary = summary.T.reset_index()
        summary.drop(columns = 'index', inplace = True)
        dev_only = summary[summary['balance']!={0:50}]
        num_queries = dev_only.shape[0]

        # getting old MRR
        original_MRR = get_MRR(dev_only['original_RR'])
        new_MRR = get_MRR(dev_only['new_RR'])

        train_only = dev_only[dev_only['query'].isin(train_queries)]
        test_only = dev_only[dev_only['query'].isin(test_queries)]

        train_original_MRR = get_MRR(train_only['original_RR'])
        test_original_MRR = get_MRR(test_only['original_RR'])
        test_new_MRR = get_MRR(test_only['new_RR'])
        train_new_MRR = get_MRR(train_only['new_RR'])
        
        logger.info(f"Number of unique queries tested: {str(num_queries)}")
        logger.info(f"Old MRR: {str(original_MRR)}")
        logger.info(f"New MRR: {str(new_MRR)}")
        if new_MRR < original_MRR:
            logger.warning("WARNING! Model did not improve MRR")
            
        summary.to_csv(os.path.join(data_dir, timestamp_filename("finetuning_results", ".csv")))
        
        ft_metadata = {
            "date_finetuned": str(date.today()),
            "data_dir": str(data_dir),
            "old_MRR": f"{str(original_MRR)} ({str(train_original_MRR)} train / {str(test_original_MRR)} test)",
            "new_MRR": f"{str(new_MRR)} ({str(train_new_MRR)} train / {str(test_new_MRR)} test)"
        }

        return ft_metadata
    
    def retrain(self, data_dir, testing_only, version):

        try:
            data = open_json("training_data.json", data_dir)
            train = data["train"]
            test = data["test"]

            del data
            gc.collect()

            if testing_only:
                logger.info("Creating smaller dataset just for testing finetuning.")
                train_keys = list(train.keys())[:10]
                test_keys = list(test.keys())[:10]
                train = {k:train[k] for k in train_keys}
                test = {k:test[k] for k in test_keys}

            ## make formatted training data
            logger.info("Formatting the inputs...")
            train_samples, df = format_inputs(train, test)
            
            ## get cosine sim before finetuning
            df["original_cos_sim"] = df["pair"].apply(lambda x: get_cos_sim(self.model, x))
            processmanager.update_status(processmanager.training, 0, 1) 
            sleep(0.1)

            ## finetune on samples
            logger.info("Starting dataloader...")
            train_dataloader = DataLoader(train_samples, shuffle=self.shuffle, batch_size=self.batch_size) #pin_memory=self.pin_memory)
            train_loss = losses.CosineSimilarityLoss(model=self.model)
            del train_samples
            gc.collect()
            logger.info("Finetuning the encoder model...")
            self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.epochs, warmup_steps=self.warmup_steps)
            processmanager.update_status(processmanager.training, 1, 0)
            logger.info("Finished finetuning the encoder model") 
            ## save model
            self.model.save(self.model_save_path)
            logger.info("Finetuned model saved to {}".format(str(self.model_save_path)))

            logger.info("*** Summarizing finetuning results ")
            ## get new cosine sim
            ft_metadata = self.summarize_results(df, data_dir)

            ## save metadata file
            ft_metadata_path = os.path.join(data_dir, timestamp_filename("finetuning_metadata", ".json"))
            with open(ft_metadata_path, "w") as outfile:
                json.dump(ft_metadata, outfile)

            logger.info("Metadata saved to {}".format(ft_metadata_path))

            evals_dir = os.path.join(self.model_save_path, "evals_gc")
            if not os.path.isdir(evals_dir):
                os.mkdir(os.path.join(evals_dir))
            ft_evals_path = os.path.join(evals_dir, timestamp_filename("finetuning_evals", ".json"))
            with open(ft_evals_path, "w") as outfile:
                json.dump(ft_metadata, outfile)
            
            logger.info("Metadata saved to {}".format(ft_evals_path))
            
            # when not testing only, save to S3
            if not testing_only:
                logger.info("Saving data to S3...")
                s3_path = os.path.join(S3_DATA_PATH, f"{version}")
                logger.info(f"****    Saving new data files to S3: {s3_path}")
                dst_path = data_dir + ".tar.gz"
                model_name = datetime.now().strftime("%Y%m%d")
                logger.info("*** Attempting to save data tar")
                utils.create_tgz_from_dir(data_dir, dst_path)
                logger.info("*** Attempting to upload data to s3")
                utils.upload(s3_path, dst_path, "data", model_name)

                logger.info("Saving model to S3...")
                dst_path = self.model_save_path + ".tar.gz"
                utils.create_tgz_from_dir(src_dir=self.model_save_path, dst_archive=dst_path)
                model_id = self.model_save_path.split('_')[1]
                logger.info(f"*** Created tgz file and saved to {dst_path}")

                S3_MODELS_PATH = "bronze/gamechanger/models"
                s3_path = os.path.join(S3_MODELS_PATH, str(version))
                utils.upload(s3_path, dst_path, "transformers", model_id)
                logger.info(f"*** Saved model to S3: {s3_path}")

            return ft_metadata
        
        except Exception as e:
            logger.warning("Could not complete finetuning")
            logger.error(e, exc_info=True)
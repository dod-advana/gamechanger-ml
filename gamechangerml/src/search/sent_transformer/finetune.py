from sentence_transformers import SentenceTransformer, InputExample, util, losses
from torch.utils.data import DataLoader
import pandas as pd
from datetime import date

from gamechangerml.src.utilities.model_helper import open_json
from gamechangerml.api.utils.logger import logger

def get_cos_sim(model, pair):

    emb1 = model.encode(pair[0])
    emb2 = model.encode(pair[1])
    cos_sim = float(util.cos_sim(emb1, emb2))
    
    return cos_sim

def format_inputs(train, test):
    '''Create input data for dataloader and df for tracking cosine sim'''

    train_samples = []
    all_data = []
    count = 0
    for i in train.keys():
        texts = [train[i]['query'], train[i]['paragraph']]
        score = float(train[i]['label'])
        inputex = InputExample(str(count), texts, score)
        train_samples.append(inputex)
        all_data.append([i, texts, score, 'train'])
        count += 1
    
    for x in test.keys():
        texts = [test[i]['query'], test[i]['paragraph']]
        score = float(test[i]['label'])
        all_data.append([i, texts, score, 'test'])

    df = pd.DataFrame(all_data, columns = ['key', 'pair', 'score', 'label'])
    
    return train_samples, df

class STFinetuner():

    def __init__(self, model, model_load_path, model_save_path, shuffle, batch_size, epochs, warmup_steps):

        if model:
            self.model = model
        else:
            self.model = SentenceTransformer(model_load_path)

        self.model_save_path = model_save_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
    
    def retrain(self, data_dir):

        data = open_json('training_data.json', data_dir)
        metadata = open_json('training_metadata.json', data_dir)
        train = data['train']
        test = data['test']
        # make formatted training data
        train_samples, df = format_inputs(train, test)
        
        # get cosine sim before finetuning
        df['original_cos_sim'] = df['pair'].apply(lambda x: get_cos_sim(self.model, x))

        ## finetune on samples
        train_dataloader = DataLoader(train_samples, shuffle=self.shuffle, batch_size=self.batch_size)
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.epochs, warmup_steps=self.warmup_steps)

        ## save model
        self.model.save(self.model_save_path)
        logger.info("Finetuned model saved to {}".format(str(self.model_save_path)))

        ## get new cosine sim
        df['new_cos_sim'] = df['pair'].apply(lambda x: get_cos_sim(self.model, x))
        df['change_cos_sim'] = df['new_cos_sim'] - df['orig_cos_sim']

        ## create training metadata
        positive_change_train = df[(df['score']==1.0) and (df['label']=='train')]['change_cos_sim'].median()
        negative_change_train = df[(df['score']==0.0) and (df['label']=='train')]['change_cos_sim'].median()
        positive_change_test = df[(df['score']==1.0) and (df['label']=='test')]['change_cos_sim'].median()
        negative_change_test = df[(df['score']==0.0) and (df['label']=='test')]['change_cos_sim'].median()

        ft_metadata = {
            "date_finetuned": str(date.today()),
            "data_dir": str(data_dir),
            "positive_change_train": positive_change_train,
            "negative_change_train": negative_change_train,
            "positive_change_test": positive_change_test,
            "negative_change_test": negative_change_test
        }

        with 
        return 
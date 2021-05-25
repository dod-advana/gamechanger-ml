import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from gamechangerml.src.search.sent_transformer.corpus import SentenceCorpus

from sentence_transformers import SentenceTransformer, models

class GCDataset(TensorDataset):
    def __init__(self, fpath, iter_len = 1_000):
        self.corp = SentenceCorpus(fpath, iter_len = iter_len)
        self.iter_len = iter_len

    def __len__(self):
        return self.iter_len

    def __getitem__(self, index):
        text_a, text_b, sim_score = self.corp._get_item_sample()
        return text_a, text_b, sim_score

def train_model(corpus_directory,
                save_path,
                pretrained_model = "msmarco-distilbert-base-v2",
                sample_count = 5_000):

    model = SentenceTransformer(pretrained_model)
    if use_gpu:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.cuda()
        else:
            print("Error")
            use_gpu = False
    dataset = GCDataset(corpus_directory, tokenizer, iter_len = sample_count)
    dataloader = DataLoader(dataset, shuffle=True, batch_size = batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

if __name__ == "__main__":
    train_model("./parsed_good", "./embeddertrained")

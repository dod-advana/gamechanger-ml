import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from gamechangerml.src.search.sent_transformer.corpus import SentenceCorpus

from sentence_transformers import SentenceTransformer, models, losses, InputExample

class GCDataset(TensorDataset):
    def __init__(self, fpath, iter_len = 1_000):
        self.corp = SentenceCorpus(fpath, iter_len = iter_len)
        self.iter_len = iter_len

    def __len__(self):
        return self.iter_len

    def __getitem__(self, index):
        print("hi")
        text_a, text_b, sim_score = self.corp._get_item_sample()
        return InputExample(texts=[text_a[0], text_b[0]], label=float(sim_score/9))

def train_model(corpus_directory,
                save_path,
                pretrained_model = "msmarco-distilbert-base-v2",
                sample_count = 10,
                use_gpu = False,
                batch_size = 4):

    model = SentenceTransformer(pretrained_model)
    if use_gpu:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.cuda()
        else:
            print("Error")
            use_gpu = False
    dataset = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
    #dataset = GCDataset(corpus_directory, iter_len = sample_count)
    dataloader = DataLoader(dataset, shuffle=True, batch_size = batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(dataloader, train_loss)], epochs=1, warmup_steps=1)

if __name__ == "__main__":
    train_model("../parsed good", "./embeddertrained")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import csv
import math

from gamechangerml.src.search.sent_transformer.corpus import SentenceCorpus

from sentence_transformers import SentenceTransformer, models, losses, InputExample
from datetime import datetime
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def train_model(save_path,
                pretrained_model = "msmarco-distilbert-base-v2",
                use_gpu = False,
                batch_size = 16):

    model = SentenceTransformer(pretrained_model)
    save_path = 'output/'+pretrained_model+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    '''if use_gpu:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.cuda()
        else:
            print("Error")
            use_gpu = False'''
    dataset = []
    with open('dataset.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            dataset.append(InputExample(texts=[row[0], row[1]], label=float(row[2])))
    testset = []
    with open('testset.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            testset.append(InputExample(texts=[row[0], row[1]], label=float(row[2])))



    dataloader = DataLoader(dataset, shuffle=True, batch_size = batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    num_epochs = 4
    warmup_steps = math.ceil(len(dataloader) * num_epochs * 0.1)
    model.fit(train_objectives=[(dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps,output_path=save_path)
    model.save(save_path)
    model = SentenceTransformer(save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(testset, name='sts-test')
    test_evaluator(model, output_path=save_path)

if __name__ == "__main__":
    train_model("./embeddertrained")

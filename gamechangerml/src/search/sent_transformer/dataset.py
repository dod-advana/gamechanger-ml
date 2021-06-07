from gamechangerml.src.search.sent_transformer.corpus import SentenceCorpus
import numpy as np
import csv

class GCDataset():
    def __init__(self, fpath, iter_len = 1_000):
        self.corp = SentenceCorpus(fpath, iter_len = iter_len)
        self.iter_len = iter_len

    def __len__(self):
        return self.iter_len

    def getitem(self):
        text_a, text_b, sim_score = self.corp._get_item_sample()
        return [text_a[0], text_b[0], float(sim_score)]


if __name__ == "__main__":
    dataset = GCDataset("../gamechanger-data-old/parsed good")
    results = []
    for i in range(800):
        results.append(dataset.getitem())


    with open("testset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

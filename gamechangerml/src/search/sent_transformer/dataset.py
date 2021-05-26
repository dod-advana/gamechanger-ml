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

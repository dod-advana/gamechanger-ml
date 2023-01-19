from transformers import BertTokenizer

class BertTokenizerCustom(object):
    def __init__(self, vocab_file=None):
        if vocab_file is None:
            vocab_file = "./assets/bert_vocab.txt"
        self.tokenizer = BertTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)

        return tokens, len(tokens)

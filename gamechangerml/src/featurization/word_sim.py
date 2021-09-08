import gensim


class WordSim:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        try:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                self.model_dir)
        except Exception as e:
            self.model = None
            print("Cannot load pretrained vector for Word Similarity")

    def tokenize(self, text: str):
        return list(gensim.utils.tokenize(text))

    def most_similiar_tokens(self, text: str, sim_thresh=0.7):
        tokens = self.tokenize(text)
        similar_tokens = {}
        for word in tokens:
            sim_words = self.model.most_similar(word)
            sim_word_thresh = [x for x in sim_words if x[1] > sim_thresh]
            similar_tokens[word] = [x[0] for x in sim_word_thresh]
        return similar_tokens

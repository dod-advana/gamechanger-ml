from gamechangerml.src.text_handling import corpus
from gamechangerml.src.text_handling.custom_stopwords import custom_stopwords
from gamechangerml.src.text_handling.process import topic_processing
from gamechangerml.api.utils import processmanager, status_updater
from gamechangerml.configs.config import TopicsConfig
from gamechangerml.src.utilities.test_utils import get_user

import gensim
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
from gensim.models.phrases import Phraser, Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.parsing.preprocessing import STOPWORDS

import os
import logging
import json
import typing as t
from pathlib import Path
import random
import nltk
from datetime import datetime

# nltk.download("averaged_perceptron_tagger")
logger = logging.getLogger("gamechanger")


class Topics(object):
    """
    TF-iDF topic model wrapper class.

    This class is written to be processing pipeline agnostic, but
    because of that be sure that you track your own pipeline for
    training and inference otherwise results may vary.

    Also note that this class will allow you to perform topic modeling
    on out of corpus documents.  Be sure to monitor the number of
    documents being processed out of corpus to ensure that the model
    still reflects the statistics of the corpus.  When this is no longer
    the case you can either retrain from scatch. TODO: Implement an
    update function to update the model on only the new documents.

    Class requirements:
        from gensim.models.tfidfmodel import TfidfModel
        from gensim import corpora
    """

    def __init__(self, directory=None, verbose=False):
        try:
            if directory is not None:
                self.load(directory)
        except Exception as e:
            logger.warning("Could not load topics model")
            logger.warning(e)

        if verbose:
            try:
                logging.basicConfig(
                    format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO,
                )
            except:
                import logging

                logging.basicConfig(
                    format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO,
                )
        else:
            try:
                logging.basicConfig(
                    format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.ERROR,
                )
            except:
                pass

    def load(self, directory):
        """
        load - class function to load the required files from a
            directory.
        Args:
            directory (str): Path to where `tfidf_dictionary.dic`
                and `tfidf.model` are located.
        Returns:
            None
        """
        dictionary_path = os.path.join(directory, "tfidf_dictionary.dic")
        tfidf_path = os.path.join(directory, "tfidf.model")
        bigrams_path = os.path.join(directory, "bigrams.phr")
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        self.tfidf = TfidfModel.load(tfidf_path)
        self.bigrams = Phraser.load(bigrams_path)

    def save(self, directory):
        """
        save - class function to save a dictionary file and the
            tfidf model file from a training.
        Args:
            directory (str): Path to where `tfidf_dictionary.dic`
                and `tfidf.model` will be saved.  Ensure that
                there are not versions of these files in `directory`
                before saving or they will be overwritten.
        Returns:
            None
        """
        dictionary_path = os.path.join(directory, "tfidf_dictionary.dic")
        tfidf_path = os.path.join(directory, "tfidf.model")
        bigrams_path = os.path.join(directory, "bigrams.phr")
        self.dictionary.save(dictionary_path)
        self.tfidf.save(tfidf_path)
        self.bigrams.save(bigrams_path)

    def stream_tokens_from_files(self, file_list: t.List[str]) -> t.Iterator[str]:
        for filepath in file_list:
            with open(filepath) as f:
                doc = json.load(f)
                raw_text = doc.get("raw_text")
                if raw_text:
                    yield gensim.utils.simple_preprocess(
                        raw_text, min_len=1, max_len=15
                    )

    def stream_topics_from_files(
        self, file_list: t.List[str], phrase_detector_model: Phrases
    ) -> t.Generator[t.List[str], None, None]:
        for filepath in file_list:
            with open(filepath) as f:
                doc = json.load(f)
                raw_text = doc.get("raw_text")

                tokenized_raw_texts = gensim.utils.simple_preprocess(
                    raw_text, min_len=1, max_len=15
                )

                # takes original tokenized_raw_texts and applies bigram/trigram phrases learned in phrase model
                phrasified_tokenized_raw_texts = phrase_detector_model[
                    tokenized_raw_texts
                ]

                ## TODO figure out how to get nltk download inside docker
                # filter for nouns using tagger
                # position_tags = nltk.pos_tag(phrasified_tokenized_raw_texts)
                # nouns = [
                #     word[0]
                #     for word in position_tags
                #     if word[1] == "NN" and len(word[0]) > 2 and word[0] not in STOPWORDS
                # ]
                # yield nouns
                yield [
                    w
                    for w in phrasified_tokenized_raw_texts
                    if len(w) > 2 and w not in STOPWORDS
                ]

    def sample_corpus_files(
        self, corpus_dir: t.Union[str, Path], sample_rate: t.Union[float, int, None]
    ) -> t.List[t.Union[str, Path]]:
        corpus_files = list(Path(corpus_dir).glob("**/*.json"))

        # random.sample()

        if sample_rate is None:
            return list(corpus_files)

        if isinstance(sample_rate, float):
            rate_int = sample_rate * 100
        elif isinstance(sample_rate, int):
            rate_int = sample_rate
        else:
            raise RuntimeError("Unknown sample rate type")

        file_list = []
        for filepath in corpus_files:
            print(filepath)
            r = random.randint(0, 100)
            if r <= rate_int:
                file_list.append(filepath)
        return file_list

    def train_from_files(self, corpus_dir, sample_rate, local_dir):
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = status_updater.StatusUpdater(
            process_key=processmanager.topics_creation, nsteps=6,
        )
        status.next_step(message="Started topics model training")

        # Generating process metadata
        user = get_user(logger)

        # sample files
        file_list = self.sample_corpus_files(corpus_dir, sample_rate)
        status.next_step(message="Corpus sampled")

        # create phrase detection from corpus
        # bigrams.phr
        phrase_detector_model = Phrases(
            self.stream_tokens_from_files(file_list),
            min_count=1,
            threshold=50,
            connector_words=ENGLISH_CONNECTOR_WORDS,
        )
        status.next_step(message="Phrase detector model created")

        # apply phrase detection to corpus
        topics_corpus_generator = self.stream_topics_from_files(
            file_list, phrase_detector_model
        )

        # create dictionary from phrases
        # tfidf_dictionary.dic
        dictionary = corpora.Dictionary(topics_corpus_generator)
        status.next_step(message="Corpus dictionary created")

        # create tfidf model from corpora dictionary
        # tfidf.model
        tfidf = TfidfModel(dictionary=dictionary)
        status.next_step(message="Topics model created")

        # set to active
        self.bigrams = phrase_detector_model
        self.dictionary = dictionary
        self.tfidf = tfidf

        # TODO
        # save locally, names follow schema
        # tar and upload
        #

        # save files
        ## Dont overwrite current models after training
        ## Will use reload to apply them

        # model_dir = model_dest

        # if not model_id:
        #     model_id = datetime.now().strftime("%Y%m%d")

        # # get model name schema
        # model_name = "qexp_" + model_id
        # model_path = utils.create_model_schema(model_dir, model_name)
        # evals = {"results": ""}
        # params = D2VConfig.MODEL_ARGS
        # try:
        #     # build ANN indices
        #     index_dir = os.path.join(model_dest, model_path)
        #     bqe.main(corpus, index_dir, **QexpConfig.MODEL_ARGS["bqe"])
        #     logger.info("-------------- Model Training Complete --------------")
        #     # Create .tgz file
        #     dst_path = index_dir + ".tar.gz"
        #     self.create_tgz_from_dir(src_dir=index_dir, dst_archive=dst_path)

        # mdir = TopicsConfig.DATA_ARGS["LOCAL_MODEL_DIR"]
        self.save(local_dir)
        status.next_step(message="Topic model saved")

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "user": user,
            "date_started": start_time,
            "date_finished": end_time,
            "corpus_name": str(corpus_dir),
        }

        return metadata

    def train(self, corpus):
        """
        train - Train a TF-iDF model on a corpus
        Args:
            corpus (iterable): An iterable where each element is a list of
                tokens.
        Returns:
            None
        """
        self.dictionary = corpora.Dictionary(corpus)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in corpus]
        self.tfidf = TfidfModel(doc_term_matrix)

    def get_topics_from_text(self, text, topn=5):
        tokens = topic_processing(text, self.bigrams)
        return self.get_topics(tokens, topn=topn)

    def get_topics(self, tokens, topn=5):
        """
        get_topics - given a tokenized text, get a list of the topn topics
            with their scores
        Args:
            tokens (list): A tokenized text list.
            topn (int): the number of topic to be returned
        Returns:
            topics (list|tuple): a list of (score, topic) pairs
        """
        doc_term_matrix = [self.dictionary.doc2bow(tokens)]
        doc_tfidf = self.tfidf[doc_term_matrix]

        word = []
        doc = doc_tfidf[0]
        for id, value in doc:
            if self.dictionary.get(id) not in custom_stopwords:
                word.append((value, self.dictionary.get(id)))
        word.sort(reverse=True)
        return word[:topn]

import logging
import os
from gamechangerml.src.paths import FEATURES_DATA_DIR

logger = logging.getLogger(__name__)


def get_word_weight(weight_file="enwiki_vocab_min200.txt", a=1e-3):
    if a <= 0.0:
        a = 1.0

    weight_file_path = os.path.join(FEATURES_DATA_DIR, weight_file)
    print("weightfilepath: ", weight_file_path)
    
    word2weight = dict()
    try:
        with open(weight_file_path) as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        logger.exception("{}: {}".format(type(e), str(e)), exc_info=True)
        raise

    N = 0
    for i in lines:
        i = i.strip()
        if len(i) > 0:
            i = i.split()
            if len(i) == 2:
                word2weight[i[0]] = float(i[1])
                N += float(i[1])

    for key, value in word2weight.items():
        word2weight[key] = a / (a + value / N)

    logger.debug("read {:,} tokens from wt_file".format(len(word2weight)))
    return word2weight

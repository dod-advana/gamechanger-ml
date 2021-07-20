"""
usage: python ner_training_data.py [-h] -s SENT_CSV -e ENTITY_CSV -o
                                   OUTPUT_TXT [-n N_SAMPLES] [-r]

create NER training data in CoNLL format

optional arguments:
  -h, --help            show this help message and exit
  -s SENT_CSV, --sentence-csv SENT_CSV
                        csv of sentences and labels
  -e ENTITY_CSV, --entity-csv ENTITY_CSV
                        csv of entities & types
  -o OUTPUT_TXT, --output-txt OUTPUT_TXT
                        output file in CoNLL format
  -n N_SAMPLES, --n-samples N_SAMPLES
                        how many samples to extract and tag (0 means get
                        everything)
  -r, --random-shuffle  randomly shuffle the sentence data
"""
import logging

import gamechangerml.src.entity.entity_mentions as em
import gamechangerml.src.text_classif.utils.classifier_utils as cu

logger = logging.getLogger(__name__)


def gen_ner_training_data(abrv_re, ent_re, entity2type, sent_dict, nlp):
    """

    Args:
        abrv_re:
        ent_re:
        entity2type:
        sent_dict:
        nlp:

    Returns:

    """
    SENT = "sentence"
    PRFX = "I-"
    for row in sent_dict:
        sentence_text = row[SENT]
        if not sentence_text.strip():
            continue
        # find the spans of each token
        doc = nlp(sentence_text)
        starts_ends = [(t.idx, t.idx + len(t.orth_) - 1) for t in doc]
        ner_labels = ["O"] * len(starts_ends)
        ner_tokens = [t.orth_ for t in doc]
        ent_spans = em.entities_spans(sentence_text, ent_re, abrv_re)

        # find token indices of an extracted entity
        for ent, ent_st_end in ent_spans:
            token_idxs = [
                idx
                for idx, tkn_st_end in enumerate(starts_ends)
                if tkn_st_end[0] >= ent_st_end[0]
                and tkn_st_end[1] <= ent_st_end[1] - 1
            ]
            for idx in token_idxs:
                if ent.lower() in entity2type:
                    ner_labels[idx] = PRFX + entity2type[ent.lower()]
                else:
                    logger.error("KeyError: {}".format(ent.lower()))
        yield ner_tokens, ner_labels


def ner_training_data(entity_csv, sentence_csv, n_samples, nlp, shuffle=False):
    """

    Args:
        entity_csv (str): name of the .csv file holding entities & types
        sentence_csv (str): name of the sentence .csv
        n_samples (int): if > 0, use everything; else up to this value
        nlp (spacy.lang.en.English): spaCy language model
        shuffle (bool): if True, randomize the order of the sentences

    Returns:

    """
    abrv_re, ent_re, entity2type = em.make_entity_re(entity_csv)
    sent_dict = cu.load_data(sentence_csv, n_samples, shuffle=shuffle)
    training_generator = gen_ner_training_data(
        abrv_re, ent_re, entity2type, sent_dict, nlp
    )
    for tokens, ner_labels in training_generator:
        logger.info(tokens)
        logger.info(ner_labels)
        logger.info("---")


if __name__ == "__main__":
    import os

    from argparse import ArgumentParser

    import gamechangerml.src.text_classif.utils.log_init as li
    from gamechangerml.src.utilities.spacy_model import get_lg_nlp

    li.initialize_logger(to_file=False, log_name="none")

    fp = "python " + os.path.split(__file__)[-1]
    parser = ArgumentParser(
        prog=fp,
        description="create NER training data in CoNLL format",
    )
    parser.add_argument(
        "-s",
        "--sentence-csv",
        dest="sent_csv",
        type=str,
        help="csv of sentences and labels",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--entity-csv",
        dest="entity_csv",
        type=str,
        help="csv of entities & types",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-txt",
        dest="output_txt",
        type=str,
        required=True,
        help="output file in CoNLL format",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        dest="n_samples",
        type=int,
        required=False,
        default=0,
        help="how many samples to extract and tag (0 means get everything)",
    )
    parser.add_argument(
        "-r",
        "--random-shuffle",
        dest="shuffle",
        action="store_true",
        default=False,
        help="randomly shuffle the sentence data",
    )
    args = parser.parse_args()

    logger.info("retrieving spaCy model")
    nlp_ = get_lg_nlp()
    logger.info("spaCy model loaded")

    ner_training_data(
        args.entity_csv, args.sent_csv, args.n_samples, nlp_, args.shuffle
    )

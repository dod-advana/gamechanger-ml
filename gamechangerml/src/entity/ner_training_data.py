import logging

import gamechangerml.src.entity.entity_mentions as em
import gamechangerml.src.text_classif.utils.classifier_utils as cu

logger = logging.getLogger(__name__)

SENT = "sentence"


def gen_ner_training_data(abrv_re, ent_re, entity2type, sent_dict, nlp):
    for row in sent_dict:
        sentence_text = row[SENT]
        if not sentence_text.strip():
            continue
        logger.info(sentence_text)
        doc = nlp(sentence_text)
        starts_ends = [(t.idx, t.idx + len(t.text) - 1) for t in doc]
        logger.info(starts_ends)
        ner_labels = ["O"] * len(starts_ends)
        ner_tokens = [t.orth_ for t in doc]
        logger.info(ner_tokens)
        ent_spans = em.entities_spans(sentence_text, ent_re, abrv_re)
        logger.info("ent spans: {}".format(ent_spans))
        for ent, ent_st_end in ent_spans:
            token_idxs = [
                idx
                for idx, tkn_st_end in enumerate(starts_ends)
                if tkn_st_end[0] >= ent_st_end[0]
                and tkn_st_end[1] <= ent_st_end[1] - 1
            ]
            logger.info("entity indexes '{}' : {}".format(ent, token_idxs))
            for idx in token_idxs:
                if ent.lower() in entity2type:
                    ner_labels[idx] = entity2type[ent.lower()]
                else:
                    logger.error("KeyError: {}".format(ent.lower()))
        yield ner_tokens, ner_labels


def ner_training_data(entity_csv, sentence_csv, n_samples, nlp):
    """

    Args:
        entity_csv:
        sentence_csv:
        n_samples:
        nlp:

    Returns:

    """
    abrv_re, ent_re, entity2type = em.make_entity_re(entity_csv)
    sent_dict = cu.load_data(sentence_csv, n_samples, shuffle=False)
    training_generator = gen_ner_training_data(
        abrv_re, ent_re, entity2type, sent_dict, nlp
    )
    for ner_tokens, ner_labels in training_generator:
        logger.info(ner_tokens)
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
        dest="output-txt",
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
        help="how many samples to extract and tag",
    )
    args = parser.parse_args()

    logger.info("retrieving spaCy model")
    nlp_ = get_lg_nlp()
    logger.info("spaCy model loaded")

    ner_training_data(args.entity_csv, args.sent_csv, args.n_samples, nlp_)

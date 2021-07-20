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

from tqdm import tqdm

import gamechangerml.src.entity.entity_mentions as em
import gamechangerml.src.text_classif.utils.classifier_utils as cu

logger = logging.getLogger(__name__)


def _wc(txt):
    return txt.count(" ") + 1


def _gen_ner_training_data(abrv_re, ent_re, entity2type, sent_dict, nlp):
    SENT = "sentence"
    I_PRFX = "I-"
    B_PRFX = "B-"
    for row in sent_dict:
        sentence_text = row[SENT]
        if not sentence_text.strip():
            continue

        doc = nlp(sentence_text)
        starts_ends = [(t.idx, t.idx + len(t.orth_) - 1) for t in doc]
        ner_labels = ["O"] * len(starts_ends)
        tokens = [t.orth_ for t in doc]
        ent_spans = em.entities_spans(sentence_text, ent_re, abrv_re)

        # find token indices of an extracted entity using the spans
        for ent, ent_st_end in ent_spans:
            ent_ntoks = _wc(ent)
            token_idxs = [
                idx
                for idx, tkn_st_end in enumerate(starts_ends)
                if tkn_st_end[0] >= ent_st_end[0]
                and tkn_st_end[1] <= ent_st_end[1] - 1
            ]
            # make CoNLL tags
            if ent_ntoks == 1:
                ner_labels[token_idxs[0]] = I_PRFX + entity2type[ent.lower()]
                continue

            ner_labels[token_idxs[0]] = B_PRFX + entity2type[ent.lower()]
            for idx in token_idxs[1:]:
                if ent.lower() in entity2type:
                    ner_labels[idx] = I_PRFX + entity2type[ent.lower()]
                else:
                    logger.error("KeyError: {}".format(ent.lower()))
        yield zip(tokens, ner_labels)


def ner_training_data(
    entity_csv, sentence_csv, n_samples, nlp, out_fp, shuffle=False
):
    """

    Args:
        entity_csv (str): name of the .csv file holding entities & types

        sentence_csv (str): name of the sentence .csv

        n_samples (int): if > 0, use everything; else up to this value

        nlp (spacy.lang.en.English): spaCy language model

        out_fp (str): where to write the resulting `.tsv` file

        shuffle (bool): if True, randomize the order of the sentences

    """
    TAB = "\t"
    NL = "\n"
    print_str = ""
    refresh = 1024
    abrv_re, ent_re, entity2type = em.make_entity_re(entity_csv)
    sent_dict = cu.load_data(sentence_csv, n_samples, shuffle=shuffle)
    training_generator = _gen_ner_training_data(
        abrv_re, ent_re, entity2type, sent_dict, nlp
    )
    count = 0
    with open(out_fp, "w") as fp:
        for zipped in tqdm(
            training_generator, total=len(sent_dict), desc="sentences"
        ):
            count += 1
            print_str += (
                NL.join([str(e[0]) + TAB + str(e[1]) for e in zipped])
                + NL
                + NL
            )
            if count > 0 and count % refresh == 0:
                fp.write(print_str)
                count = 0
                print_str = ""
        if print_str:
            fp.write(print_str[:-1])
    logger.info("output written to : {}".format(out_fp))


if __name__ == "__main__":
    import os

    from argparse import ArgumentParser

    import gamechangerml.src.text_classif.utils.log_init as li
    from gamechangerml.src.utilities.spacy_model import get_lg_nlp

    li.initialize_logger(to_file=False, log_name="none")

    fp_ = "python " + os.path.split(__file__)[-1]
    parser = ArgumentParser(
        prog=fp_,
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
        "--output-tsv",
        dest="output_tsv",
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
        args.entity_csv,
        args.sent_csv,
        args.n_samples,
        nlp_,
        args.output_tsv,
        args.shuffle,
    )

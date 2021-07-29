"""
usage: python ner_training_data.py [-h] -s SENT_CSV -e ENTITY_CSV
                                   [-n N_SAMPLES] [-r] [-p {tab,space}]
                                   [-x T_SPLIT] [--min-tokens MIN_TOKENS]
                                   [--max-tokens MAX_TOKENS]

Create NER training data in CoNLL format

optional arguments:
  -h, --help            show this help message and exit
  -s SENT_CSV, --sentence-csv SENT_CSV
                        csv of input sentences and labels
  -e ENTITY_CSV, --entity-csv ENTITY_CSV
                        csv of entities & types
  -n N_SAMPLES, --n-samples N_SAMPLES
                        how many samples to extract and tag (0 means get
                        everything)
  -r, --shuffle         randomly shuffle the sentence data
  -p {tab,space}, --separator {tab,space}
                        token <-> label separator, default is 'space'
  -x T_SPLIT, --train-split T_SPLIT
                        training split; dev, val are evenly split from 1 -
                        t_split
  --min-tokens MIN_TOKENS
                        minimum number of tokens in a sentence
  --max-tokens MAX_TOKENS
                        maximum number of tokens in a sentence
"""
import logging
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import gamechangerml.src.entity.entity_mentions as em
import gamechangerml.src.text_classif.utils.classifier_utils as cu

logger = logging.getLogger(__name__)

SENT = "sentence"


def wc(txt):
    return txt.count(" ") + 1


# TODO exclude abbreviations if it's part of an entity? e.g.,
# TODO "The Director, DLA shall:". This will be tagged as
# TODO  O GCPER GCPER GCORG-ABBRV O O
def _gen_ner_conll_tags(abbrv_re, ent_re, entity2type, sent_dict, nlp):
    I_PRFX = "I-"
    B_PRFX = "B-"
    OH = "O"

    for row in sent_dict:
        sentence_text = row[SENT]
        if not sentence_text.strip():
            continue

        doc = nlp(sentence_text)
        starts_ends = [(t.idx, t.idx + len(t.orth_) - 1) for t in doc]
        ner_labels = [OH] * len(starts_ends)
        tokens = [t.orth_ for t in doc]
        ent_spans = em.entities_spans(sentence_text, ent_re, abbrv_re)

        # find token indices of an extracted entity using their spans;
        # create CoNLL tags
        for ent, ent_st_end in ent_spans:
            token_idxs = [
                idx
                for idx, tkn_st_end in enumerate(starts_ends)
                if tkn_st_end[0] >= ent_st_end[0]
                and tkn_st_end[1] <= ent_st_end[1] - 1
            ]
            if not token_idxs:
                continue
            if wc(ent) == 1:
                ner_labels[token_idxs[0]] = I_PRFX + entity2type[ent.lower()]
                continue

            ner_labels[token_idxs[0]] = B_PRFX + entity2type[ent.lower()]
            for idx in token_idxs[1:]:
                if ent.lower() in entity2type:
                    ner_labels[idx] = I_PRFX + entity2type[ent.lower()]
                else:
                    logger.error("KeyError: {}".format(ent.lower()))
        unique_labels = set(ner_labels)
        yield zip(tokens, ner_labels), unique_labels


def ner_training_data(
    entity_csv,
    sentence_csv,
    n_samples,
    nlp,
    sep,
    out_fp,
    shuffle,
    abbrv_re=None,
    entity_re=None,
    entity2type=None,
    min_tokens=None,
    max_tokens=None,
):
    """
    Create NER training data in CoNLL-2003 format. For more information on
    the tagging conventions, see https://huggingface.co/datasets/conll2003.

    Args:
        entity_csv (str): name of the .csv file holding entities & types

        sentence_csv (str): name of the sentence .csv

        n_samples (int): if > 0, use everything; else up to this value

        nlp (spacy.lang.en.English): spaCy language model

        sep (str): separator between entity & label

        out_fp (str): where to write the resulting `.tsv` file

        shuffle (bool): if True, randomize the order of the sentences

        abbrv_re (SRE_Pattern): compiled regular expression; optional

        entity_re (SRE_Pattern): compiled regular expression; optional

        entity2type (dict): map of an entity to its type, e.g.,
            GCORG, GCPER, etc; optional.

        min_tokens (int): minimum number of tokens in a sentence (unused)

        max_tokens (int): maximum number of tokens in a sentence (unused)
    """
    if sep == "space":
        SEP = " "
    elif sep == "tab":
        SEP = "\t"
    else:
        msg = "unrecognized value for `sep`, got {} ".format(sep)
        msg += "using space separator"
        logger.warning("unrecognized value for `sep`, got {}".format(sep))
        SEP = " "

    NL = "\n"
    EMPTYSTR = ""
    print_str = EMPTYSTR

    if None in (abbrv_re, entity_re, entity2type):
        abbrv_re, entity_re, entity2type = em.make_entity_re(entity_csv)

    sent_list = cu.load_data(sentence_csv, n_samples, shuffle=shuffle)

    ent_sents = [
        row
        for row in sent_list
        if em.contains_entity(row[SENT], entity_re, abbrv_re)
    ]
    if not ent_sents:
        logger.warning("no entities discovered in the input...")

    all_tokens = [wc(row[SENT]) for row in sent_list]
    avg_tokens = sum(all_tokens) / len(sent_list)
    logger.info("            num sentences : {:>5,d}".format(len(sent_list)))
    logger.info("num sentences w/ entities : {:>5,d}".format(len(ent_sents)))
    logger.info("               num tokens : {:>5,d}".format(sum(all_tokens)))
    logger.info("    min tokens / sentence : {:>5d}".format(min(all_tokens)))
    logger.info("    max tokens / sentence : {:>5d}".format(max(all_tokens)))
    logger.info("    avg tokens / sentence : {:>5.2f}".format(avg_tokens))

    random.shuffle(ent_sents)

    training_generator = _gen_ner_conll_tags(
        abbrv_re, entity_re, entity2type, ent_sents, nlp
    )
    labels = set()
    count = 0
    with open(out_fp, "w") as fp:
        for zipped, unique_labels in tqdm(
            training_generator, total=len(ent_sents), desc="sentence"
        ):
            labels = labels.union(unique_labels)
            count += 1
            print_str += (
                NL.join([str(e[0]) + SEP + str(e[1]) for e in zipped])
                + NL
                + NL
            )
            fp.write(print_str)
        if print_str:
            fp.write(print_str[:-1])

    logger.info("output written to : {}".format(out_fp))


def main(entity_csv, sentence_csv, n_samples, nlp, sep, shuffle, t_split,
         min_tokens=None, max_tokens=None):
    """
    This creates CoNLL-formatted data for use in the NER model. Three files
    are created `train.txt.tmp`, `dev.txt.tmp`, and `test.txt.tmp` in the
    same director as `sentence_csv`.

    Prior to loading these for training, these files will be run through
    `preprocess` to insure the max sequence length is respected by the
    model's tokenizer.

    Args:
        entity_csv (str): csv of entities & types;
            see `entity/aux_data/entities.csv`

        sentence_csv (str): randomly chosen sentences in format
            src,label,sentence

        n_samples (int): total number of samples desired

        nlp (spacy.lang.en.English): spaCy language model

        sep (str): separate between a token and its type

        shuffle (bool): If True, randomize the sentence data

        t_split (float): fraction used for training data, e.g., 0.80; dev
            and test data are split as (1 - t_split) / 2

        min_tokens (int): minimum number of tokens in a sentence

        max_tokens (int): maximum number of tokens in a sentence
    """
    if not os.path.isfile(sentence_csv):
        raise FileExistsError("no sentence_csv; got {}".format(sentence_csv))
    abbrv_re, entity_re, entity2type = em.make_entity_re(entity_csv)

    in_path, _ = os.path.split(sentence_csv)
    sent_fnames = (
        os.path.join(in_path, "train_sent.csv"),
        os.path.join(in_path, "dev_sent.csv"),
        os.path.join(in_path, "test_sent.csv"),
    )
    output_names = [p.replace("_sent.csv", ".txt.tmp") for p in sent_fnames]

    df = pd.read_csv(sentence_csv, delimiter=",", header=None)
    logger.info("examples : {:>4,d}".format(len(df)))

    # subset the data?
    if n_samples > 0:
        df = df.head(n_samples)

    train, dev_test = train_test_split(df, train_size=t_split)
    dev, test = train_test_split(dev_test, train_size=0.50)

    # save intermediate output for now
    for idx, df in enumerate((train, dev, test)):
        df.to_csv(sent_fnames[idx], header=False, index=False, sep=",")
        fn_ = os.path.split(sent_fnames[idx])[-1]
        logger.info("samples {:>5,d} : {:>14s}".format(len(df), fn_))

    for idx in range(3):
        ner_training_data(
            entity_csv,
            sent_fnames[idx],
            n_samples,
            nlp,
            sep,
            output_names[idx],
            shuffle,
            abbrv_re=abbrv_re,
            entity_re=entity_re,
            entity2type=entity2type,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    import gamechangerml.src.text_classif.utils.log_init as li
    from gamechangerml.src.utilities.spacy_model import get_lg_nlp

    li.initialize_logger(to_file=False, log_name="none")

    parser = ArgumentParser(
        prog="python " + os.path.split(__file__)[-1],
        description="Create NER training data in CoNLL format",
    )
    parser.add_argument(
        "-s",
        "--sentence-csv",
        dest="sent_csv",
        type=str,
        help="csv of input sentences and labels",
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
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=False,
        help="randomly shuffle the sentence data",
    )
    parser.add_argument(
        "-p",
        "--separator",
        dest="sep",
        type=str,
        choices=["tab", "space"],
        default="space",
        help="token <-> label separator, default is 'space'",
    )
    parser.add_argument(
        "-x",
        "--train-split",
        dest="t_split",
        type=float,
        default=0.80,
        help="training split; dev, val are evenly split from 1 - t_split",
    )
    parser.add_argument(
        "--min-tokens",
        dest="min_tokens",
        type=int,
        default=0,
        help="minimum number of tokens in a sentence",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=500,
        help="maximum number of tokens in a sentence",
    )
    args = parser.parse_args()

    logger.info("retrieving spaCy model")
    nlp_ = get_lg_nlp()
    logger.info("spaCy model loaded")

    main(args.entity_csv, args.sent_csv, args.n_samples, nlp_, args.sep,
         args.shuffle, args.t_split, args.max_tokens)

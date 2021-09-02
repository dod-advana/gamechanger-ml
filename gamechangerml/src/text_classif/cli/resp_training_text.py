"""
usage: python table.py [-h] -i INPUT_DIR -a AGENCIES_FILE -o OUTPUT [-g GLOB]
                       -e ENTITY_CSV -d DROP_PROB -m MIN_TOKENS

Extracts responsibility statements from policy documents

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        corpus directory
  -a AGENCIES_FILE, --agencies-file AGENCIES_FILE
                        the magic agencies file
  -o OUTPUT, --output OUTPUT
                        name of the output file (.csv)
  -g GLOB, --glob GLOB  file glob to use in extracting from input_dir
  -e ENTITY_CSV, --entity-csv ENTITY_CSV
                        csv containing entities, types
  -d DROP_PROB, --drop-zero-prob DROP_PROB
                        fraction of negative examples to retain
  -m MIN_TOKENS, --min-tokens MIN_TOKENS
                        minimum tokens for a sentence
"""
import logging
import re

import pandas as pd
from nltk.tokenize import sent_tokenize
import numpy as np

import gamechangerml.src.utilities.spacy_model as spacy_m
from gamechangerml.src.featurization.table import Table
import gamechangerml.src.entity.entity_mentions as em
import gamechangerml.src.text_classif.utils.classifier_utils as cu

logger = logging.getLogger(__name__)


def wc(text):
    return text.count(" ") + 1


class ExtractRespText(Table):
    def __init__(
        self,
        input_dir,
        output,
        spacy_model,
        agency_file,
        glob,
        entity_csv,
        drop_prob,
    ):
        super(ExtractRespText, self).__init__(
            input_dir, output, spacy_model, agency_file, glob, True
        )
        self.drop_prob = drop_prob
        logger.info("input dir : {}".format(input_dir))
        self.train_df = pd.DataFrame(columns=["source", "label", "text"])

        # matches 3.2.3., etc. at the start of the text
        self.dd_re = re.compile("(^\\d\\..*?\\d+\\. )")
        self.abbrv_re, self.entity_re, _ = em.make_entity_re(entity_csv)
        self.COLON = ":"
        self.kw = "shall"
        self.kw_re = r"\b" + self.kw + r"\b"
        self.kw_colon = self.kw + self.COLON
        self.resp = "RESPONSIBILITIES"
        self.SR_LABEL = 2
        self.NR_LABEL = 0
        self.sents = None

    def scrubber(self, txt):
        txt = re.sub("[\\n\\t\\r]+", " ", txt)
        txt = re.sub("\\s{2,}", " ", txt).strip()
        mobj = self.dd_re.search(txt)
        if mobj:
            txt = txt.replace(mobj.group(1), "")
        return txt.strip()

    def entity_list(self, text):
        entity_list = em.entity_list(text, self.entity_re, self.abbrv_re)
        entity_list = [e for e in entity_list if e]
        return entity_list

    def extract_positive(self):
        for tmp_df, fname in self.extract_section(self.input_dir):
            if self.SR_LABEL in tmp_df:
                tmp_df = tmp_df[2].drop_duplicates()
                pos_ex = [self.scrubber(txt) for txt in tmp_df.tolist()]
                pos_ex = [txt for txt in pos_ex if txt]
                yield pos_ex, fname, self.raw_text

    def raw_text2sentences(self, raw_text, min_tokens):
        self.sents = [
            self.scrubber(sent)
            for sent in sent_tokenize(raw_text)
            if wc(sent) > min_tokens
        ]
        return self.sents

    def extract_neg_in_doc(self, raw_text, min_tokens):
        neg_sentences = set()
        if self.resp in raw_text:
            if self.drop_prob > np.random.uniform(0.0, 1.0):
                return neg_sentences
            prev_text = raw_text.split(self.resp)[0]
            if prev_text is not None:
                sents = self.raw_text2sentences(prev_text, min_tokens)
                for s in sents:
                    neg_sentences.add(s)
        return list(neg_sentences)

    def extract_standalone(self, sentences):
        """
        This extracts standalone statements of responsibility of the form
        `...<ENTITY>...shall...` and labels this as 2. This will allow
        the entity linking to function with minimal change.

        Statements of the form `...<ENTITY>...shall:` are labeled 0. The
        original extraction does not return statements of this form. This
        slightly increase the number of negative samples.

        Args:
            sentences (list): sentences to label

        Yields:
            tuple(str, int): sentence, label

        """
        for sent in sentences:
            entity_list = self.entity_list(sent)
            if not entity_list:  # no entities - bail
                continue
            mobj_kw = re.search(self.kw_re, sent)
            if mobj_kw is None:  # no keyword - bail
                continue
            if self.kw_colon in sent:  # start of enumerated responsibilities
                yield sent, self.NR_LABEL
            elif self.COLON not in sent:  # "... <ENTITY> ... shall ...
                logger.debug("--> 2 {} {}".format(sent, entity_list))
                yield sent, self.SR_LABEL

    def extract_standalone_resp(self, sentences):
        for sent, label in self.extract_standalone(sentences):
            yield sent, label

    def _append_df(self, source, label, texts):
        for txt in texts:
            if not txt:
                continue
            new_row = {
                "source": source,
                "label": label,
                "text": txt,
            }
            self.train_df = self.train_df.append(new_row, ignore_index=True)

    def extract_pos_neg(self, min_tokens):
        total_pos_1 = 0
        total_pos_2 = 0
        total_neg = 0
        count = 0
        for pos_ex, fname, raw_text in self.extract_positive():
            count += 1
            stand_alone = list()
            try:
                if self.resp in raw_text:
                    resp_text, _ = self.get_section(raw_text, fname)
                    resp_sentences = self.raw_text2sentences(
                        resp_text, min_tokens
                    )

                    for sent, label in self.extract_standalone_resp(
                        resp_sentences
                    ):
                        self._append_df(fname, label, [sent])
                        if label == 2:
                            stand_alone.append(sent)
                            total_pos_2 += 1

                self._append_df(fname, 1, pos_ex)
                total_pos_1 += len(pos_ex)

                neg_ex = self.extract_neg_in_doc(
                    raw_text, min_tokens=min_tokens
                )
                # belt and braces for now
                neg_ex = [n for n in neg_ex if n not in stand_alone]
                total_neg += len(neg_ex)
                self._append_df(fname, 0, neg_ex)

                logger.info(
                    "{:>35s} : {:5,d} + {:5,d} ++ {:7,d} (so far) -".format(
                        fname, total_pos_1, total_pos_2, total_neg
                    )
                )
            except ValueError as e:
                logger.exception("offending file name : {}".format(fname))
                logger.exception("{}: {}".format(type(e), str(e)))
                pass
        logger.info("     total documents : {:>7,d}".format(count))
        logger.info(
            "  training data size : {:>7,d}".format(len(self.train_df))
        )
        logger.info("negative samples (0) : {:>7,d}".format(total_neg))
        logger.info("positive samples (1) : {:>7,d}".format(total_pos_1))
        logger.info("positive samples (2) : {:>7,d}".format(total_pos_2))
        no_resp_docs = "\n".join(self.no_resp_docs)
        logger.info("no responsibilities : {}".format(no_resp_docs))


def main(
    input_dir,
    output,
    spacy_model,
    agencies_file,
    glob,
    entity_csv,
    drop_prob,
    min_tokens,
):
    extractor = ExtractRespText(
        input_dir,
        output,
        spacy_model,
        agencies_file,
        glob,
        entity_csv,
        drop_prob,
    )

    extractor.extract_pos_neg(min_tokens=min_tokens)
    logger.info("train df : {:>7,d}".format(len(extractor.train_df)))

    train_df = extractor.train_df.drop_duplicates().reset_index(drop=True)
    logger.info("train df de-dup : {:>7,d}".format(len(train_df)))

    train_df.to_csv(output, index=False, header=False, doublequote=True)
    logger.info("output written to : {}".format(args.output))
    logger.info(
        "       total time : {:}".format(cu.format_time(time() - start))
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    import os
    import sys
    from time import time
    import warnings

    log_fmt = (
        "[%(asctime)s %(levelname)-8s], [%(filename)s:%(lineno)s - "
        + "%(funcName)s()], %(message)s"
    )
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    desc = "Extracts responsibility statements from policy documents"
    parser = ArgumentParser(
        prog="python resp_training_text.py", description=desc
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        type=str,
        required=True,
        help="corpus directory",
    )
    parser.add_argument(
        "-a",
        "--agencies-file",
        dest="agencies_file",
        type=str,
        required=True,
        help="the magic agencies file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        help="name of the output file (.csv)",
    )
    parser.add_argument(
        "-g",
        "--glob",
        dest="glob",
        type=str,
        default="DoDD*.json",
        help="file glob to use in extracting from input_dir",
    )
    parser.add_argument(
        "-e",
        "--entity-csv",
        dest="entity_csv",
        type=str,
        required=True,
        help="csv containing entities, types",
    )
    parser.add_argument(
        "-d",
        "--drop-zero-prob",
        dest="drop_prob",
        type=float,
        required=True,
        help="fraction of negative examples to retain",
    )
    parser.add_argument(
        "-m",
        "--min-tokens",
        dest="min_tokens",
        type=int,
        required=True,
        help="minimum tokens for a sentence",
    )
    start = time()
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        warnings.warn("no input_dir; got {}".format(args.input_dir))
        sys.exit(1)
    if not os.path.isfile(args.agencies_file):
        warnings.warn("no agencies_file; got {}".format(args.agencies_file))
        sys.exit(1)
    if not os.path.isfile(args.entity_csv):
        warnings.warn("no entity_csv; got {}".format(args.entity_csv))
        sys.exit(1)
    if not 0.0 <= args.drop_prob <= 1.0:
        warnings.warn(
            "invalid drop probability; got {}".format(args.drop_prob)
        )
        sys.exit(1)
    if os.path.isfile(args.output):
        msg = "output file already exists. Please delete or rename {}".format(
            args.output
        )
        warnings.warn(msg)
        sys.exit(1)

    logger.info("loading spaCy")
    spacy_model_ = spacy_m.get_lg_vectors()
    logger.info("spaCy loaded...")

    extractor_ = ExtractRespText(
        args.input_dir,
        args.output,
        spacy_model_,
        args.agencies_file,
        args.glob,
        args.entity_csv,
        args.drop_prob,
    )

    extractor_.extract_pos_neg(min_tokens=args.min_tokens)
    logger.info("train df : {:>7,d}".format(len(extractor_.train_df)))

    # not strictly necessary
    train_df_ = extractor_.train_df.drop_duplicates().reset_index(drop=True)
    logger.info("train df de-dup : {:>7,d}".format(len(train_df_)))

    train_df_.to_csv(args.output, index=False, header=False, doublequote=True)
    logger.info("output written to : {}".format(args.output))
    logger.info(
        "       total time : {:}".format(cu.format_time(time() - start))
    )

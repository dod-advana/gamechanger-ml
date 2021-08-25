"""
usage: python table.py [-h] -i INPUT_DIR -a AGENCIES_FILE -o OUTPUT [-g GLOB]

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
"""
import logging
import re

import pandas as pd
from nltk.tokenize import sent_tokenize

import gamechangerml.src.utilities.spacy_model as spacy_m
from gamechangerml.src.featurization.table import Table
import gamechangerml.src.entity.entity_mentions as em

logger = logging.getLogger(__name__)


def wc(text):
    return text.count(" ") + 1


class ExtractRespText(Table):
    def __init__(
        self, input_dir, output, spacy_model, agency_file, glob, entity_csv
    ):
        super(ExtractRespText, self).__init__(
            input_dir, output, spacy_model, agency_file, glob, True
        )
        logger.info("input dir : {}".format(input_dir))
        self.train_df = pd.DataFrame(columns=["source", "label", "text"])

        # matches 3.2.3., etc. at the start of the text
        self.dd_re = re.compile("(^\\d\\..*?\\d+\\. )")
        self.abbrv_re, self.entity_re, _ = em.make_entity_re(entity_csv)
        self.COLON = ":"
        self.kw = "shall"
        self.kw_colon = self.kw + self.COLON
        self.resp = "RESPONSIBILITIES"
        self.TWO = 2
        self.sents = None

    def scrubber(self, txt):
        txt = re.sub("[\\n\\t\\r]+", " ", txt)
        txt = re.sub("\\s{2,}", " ", txt).strip()
        mobj = self.dd_re.search(txt)
        if mobj:
            txt = txt.replace(mobj.group(1), "")
        return txt.strip()

    def contains_entity(self, text):
        return em.contains_entity(text, self.entity_re, self.abbrv_re)

    def extract_positive(self):
        for tmp_df, fname in self.extract_section(self.input_dir):
            if self.TWO in tmp_df:
                tmp_df = tmp_df[2].drop_duplicates()
                pos_ex = [self.scrubber(txt) for txt in tmp_df.tolist()]
                pos_ex = [txt for txt in pos_ex if txt]
                yield pos_ex, fname, self.raw_text

    def raw_text2sentences(self, raw_text, min_len):
        self.sents = [
            self.scrubber(sent)
            for sent in sent_tokenize(raw_text)
            if len(sent) > min_len
        ]
        return self.sents

    def extract_neg_in_doc(self, raw_text, min_len):
        neg_sentences = list()
        if self.resp in raw_text:
            prev_text = raw_text.split(self.resp)[0]
            if prev_text is not None:
                sents = self.raw_text2sentences(raw_text, min_len)
                neg_sentences.extend(sents)
        return neg_sentences

    def extract_standalone(self, sentences):
        """
        This extracts standalone statements of responsibility of the form
        `...<ENTITY>...shall...` and labels this as 2. This will allow
        the entity linking to function with minimal change.

        Statements of the form `...<ENTITY>...shall:` are labeled 0. The
        original extraction does not return statements of this form. This
        will increase the number of negative samples.

        Args:
            sentences (list): sentences to label

        Yields:
            tuple(str, int): sentence, label

        """
        for sent in sentences:
            entity_list = self.contains_entity(sent)
            if not entity_list or self.kw not in sent:
                continue
            elif self.kw_colon in sent:  # start of enumerated responsibilities
                yield sent, 0
            elif self.COLON not in sent:  # ...<ENTITY>...shall...
                yield sent, 2

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

    def extract_pos_neg(self, min_len):
        total_pos_1 = 0
        total_pos_2 = 0
        total_neg = 0
        for pos_ex, fname, raw_text in self.extract_positive():
            stand_alone = list()
            try:
                if self.resp in raw_text:
                    resp_text, _ = self.get_section(raw_text, fname)
                    resp_sentences = self.raw_text2sentences(
                        resp_text, min_len
                    )
                    neg_sentences = raw_text.split(self.resp)[0]

                    for sent, label in self.extract_standalone_resp(
                        resp_sentences
                    ):
                        self._append_df(fname, label, [sent])
                        if label == 2:
                            stand_alone.append(sent)
                            total_pos_2 += 1
                    for sent, label in self.extract_standalone_resp(
                        neg_sentences
                    ):
                        if label == 2:
                            total_pos_2 += 1
                            stand_alone.append(sent)
                        self._append_df(fname, label, [sent])

                pos_examples = [p for p in pos_ex if p not in stand_alone]
                self._append_df(fname, 1, pos_examples)
                total_pos_1 += len(pos_examples)

                neg_ex = self.extract_neg_in_doc(raw_text, min_len=min_len)
                neg_examples = [n for n in neg_ex if n not in stand_alone]
                total_neg += len(neg_examples)
                self._append_df(fname, 0, neg_examples)

                logger.info(
                       "{:>35s} : {:3d} +, {:3d} ++, {:3d} -".format(
                           fname, total_pos_1, total_pos_2, total_neg
                       )
                    )
            except ValueError as e:
                logger.exception("offending file name : {}".format(fname))
                logger.exception("{}: {}".format(type(e), str(e)))
                pass
        logger.info("negative samples (0) : {:>6,d}".format(total_neg))
        logger.info("positive samples (1) : {:>6,d}".format(total_pos_1))
        logger.info("positive samples (2) : {:>6,d}".format(total_pos_2))
        logger.info("               total : {:>6,d}".format(len(self.train_df)))
        no_resp_docs = "\n".join(self.no_resp_docs)
        logger.info("no responsibilities : {}".format(no_resp_docs))


if __name__ == "__main__":
    from argparse import ArgumentParser

    log_fmt = (
        "[%(asctime)s %(levelname)-8s], [%(filename)s:%(lineno)s - "
        + "%(funcName)s()], %(message)s"
    )
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    desc = "Extracts responsibility statements from policy documents"
    parser = ArgumentParser(prog="python table.py", description=desc)

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

    args = parser.parse_args()

    logger.info("loading spaCy")
    spacy_model_ = spacy_m.get_lg_vectors()
    logger.info("spaCy loaded...")
    # logger.info(spacy_model_.pipe_names)

    extract_obj = ExtractRespText(
        args.input_dir,
        args.output,
        spacy_model_,
        args.agencies_file,
        args.glob,
        args.entity_csv,
    )

    extract_obj.extract_pos_neg(min_len=1)
    logger.info(extract_obj.train_df.head())
    extract_obj.train_df.to_csv(
        args.output, index=False, header=False, doublequote=True
    )

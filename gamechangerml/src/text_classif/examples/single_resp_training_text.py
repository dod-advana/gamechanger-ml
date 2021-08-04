import pandas as pd
import re
import logging
import json
import os
import spacy
import fnmatch

from nltk.tokenize import sent_tokenize

from gamechangerml.src.text_classif.utils.entity_link import EntityLink
from gamechangerml.src.text_classif.utils.entity_lookup import ContainsEntity
from gamechangerml.src.featurization.table import Table

logger = logging.getLogger(__name__)


#TODO: Use synthetic data for negative examples?
#TODO: Add feature to allow for multiple verbs at command line instead of hard-coded


class SingleRespTrain(Table):
    def __init__(self, input_dir, output, spacy_model, agency_file, glob):
        super(SingleRespTrain, self).__init__(
            input_dir, output, spacy_model, agency_file, glob, True
        )
        logger.info('input dir : {}'.format(input_dir))
        self.train_df = pd.DataFrame(columns=['source', 'label', 'text'])
        
        self.dd_re = re.compile("(^\\d\\..*?\\d+\\. )")
        self.kw = "shall"
        self.resp = "RESPONSIBILITIES"
        self.contains_entity = ContainsEntity()
        self.train_df = pd.DataFrame(columns=['source', 'label', 'text'])
        self.resp_verbs = ['shall']
        self.agencies = pd.read_csv(agency_file)
        
    def scrubber(self, txt):
        txt = re.sub("[\\n\\t\\r]+", " ", txt)
        txt = re.sub("\\s{2,}", " ", txt).strip()
        mobj = self.dd_re.search(txt)
        if mobj:
            txt = txt.replace(mobj.group(1), "")
        return txt.strip()
        
    def extract_single(self, input_dir):
        temp_df = pd.DataFrame(columns=['source', 'label', 'text'])
        for file in sorted(os.listdir(input_dir)):
            if not fnmatch.fnmatch(file, self.glob):
                continue
            with open(os.path.join(input_dir, file)) as f_in:
                try:
                    self.doc_dict = json.load(f_in)
                except json.JSONDecodeError:
                    logger.warning("could not decode `{}`".format(file))
                    continue
            file = self.doc_dict["filename"]
            text = self.doc_dict["raw_text"]
            self.raw_text = text
            if self.resp in text:
                resp_text, entity = self.get_section(text, file)
            else:
                continue
            tokenized = sent_tokenize(resp_text)           
            for i in tokenized:
                for j in self.resp_verbs:
                    if j in i:
                        if ":" not in i:
                            single_resp = self.scrubber(i)
                            if len(single_resp) > 100:
                                temp = {'source': file, 'text': single_resp, 'label':1}
                            else:
                                temp = {'source': file, 'text': single_resp, 'label':0}
                            #TODO: replace with Chris's NER when available
                            # for i, row in self.agencies.iterrows():
                            #     if row['Agency_Aliases'] in single_resp:
                            #         temp = {'source': file, 'text': single_resp, 'label':1}
                            #     elif row['Agency_Name'] in single_resp:
                            #         temp = {'source': file, 'text': single_resp, 'label':1}
                            #     else:
                            #         temp = {'source': file, 'text': single_resp, 'label':0}
                            temp_df = temp_df.append(temp, ignore_index=True)
            logger.info(
                "{:>25s} : {:>3,d}".format(
                    self.doc_dict["filename"], len(temp_df)
                )
            )
            yield temp_df, file

    def extract_header(self, input_dir):
        temp_df = pd.DataFrame(columns=['source', 'label', 'text'])
        for file in sorted(os.listdir(input_dir)):
            if not fnmatch.fnmatch(file, self.glob):
                continue
            with open(os.path.join(input_dir, file)) as f_in:
                try:
                    self.doc_dict = json.load(f_in)
                except json.JSONDecodeError:
                    logger.warning("could not decode `{}`".format(file))
                    continue
            file = self.doc_dict["filename"]
            text = self.doc_dict["raw_text"]
            self.raw_text = text
            if self.resp in text:
                resp_text, entity = self.get_section(text, file)
            else:
                continue
            for i in resp_text.split('.'):
                for j in i.split('\n'):
                    if "shall:" in j:
                        if len(j) > 10:
                            temp = {'source': file, 'text': j, 'label':2}
                            temp_df = temp_df.append(temp, ignore_index=True) 
                        else:
                            temp = {'source': file, 'text': j, 'label':0}
                            temp_df = temp_df.append(temp, ignore_index=True)          
            # temp_df = temp_df.drop_duplicates(subset=['source', 'text']).reset_index(drop=True)
            yield temp_df, file
    
    def process_all(self):
        for tmp_df, fname in self.extract_single(self.input_dir):
            self.train_df = self.train_df.append(tmp_df, ignore_index=True)
        
        for tmp_df, fname in self.extract_header(self.input_dir):
            self.train_df = self.train_df.append(tmp_df, ignore_index=True)
        
        tmp = self.train_df.drop_duplicates(subset=['source', 'text']).sort_values(by=['source']).reset_index(drop=True)
        # tmp = self.train_df.sort_values(by=['source']).reset_index(drop=True)
        return tmp


if __name__ == "__main__":
    from argparse import ArgumentParser

    desc = "Extracts single-sentence responsibility statements from policy documents"
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
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        help="name of the output file (.csv)",
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
        "-g",
        "--glob",
        dest="glob",
        type=str,
        default="DoDD*.json",
        help="file glob to use in extracting from input_dir",
    )

    args = parser.parse_args()

    logger.info("loading spaCy")
    spacy_model_ = spacy.load('en_core_web_lg')
    logger.info("spaCy loaded...")

    table_obj = SingleRespTrain(
        args.input_dir,
        args.output,
        spacy_model_,
        args.agencies_file,
        args.glob,
    )

    output_file = table_obj.process_all()
    #TODO: move the csv call to process_all()
    output_file.to_csv(args.output, index=False, header=False, doublequote=True)
    logger.info("training data extracted")

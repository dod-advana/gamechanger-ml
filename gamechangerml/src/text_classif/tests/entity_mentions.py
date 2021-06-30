"""
usage: python entity_mentions.py [-h] -i INPUT_PATH -e ENTITY_FILE -o
                                 OUTPUT_JSON -g GLOB

outputs counts of entity mentions in each document

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH
                        corpus path
  -e ENTITY_FILE, --entity-file ENTITY_FILE
                        list of entities with their abbreviations
  -o OUTPUT_JSON, --output-json OUTPUT_JSON
                        output path for .csv files
  -g GLOB, --glob GLOB  file pattern to match
"""
import logging
import re
import time
from collections import defaultdict

import gamechangerml.src.text_classif.utils.classifier_utils as cu

logger = logging.getLogger(__name__)


def get_entity(orgs_file):
    abbrvs = set()
    entities = set()

    with open(orgs_file) as f:
        entity_list = f.readlines()
    for line in entity_list:
        if line.startswith("#"):
            continue
        line = line.strip()
        if "(" in line:
            entity, abbrv = line.split("(", maxsplit=1)
        else:
            entity = line
            abbrv = None
        entities.add(entity)
        if abbrv and abbrv.endswith(")"):
            abbrvs.add(abbrv[:-1])

    entities = list(entities)
    abbrvs = list(abbrvs)
    logger.info("num entities : {}".format(len(entities)))
    logger.info(" num abbrevs : {}".format(len(abbrvs)))

    entities.sort(key=lambda s: len(s))
    abbrvs.sort(key=lambda s: len(s))

    entity_re = "|".join([re.escape(e.strip()) for e in entities])
    entity_re = re.compile(entity_re, re.I)

    abbrv_re = "|".join(([re.escape(a.strip()) for a in abbrvs]))
    abbrv_re = re.compile(abbrv_re)
    return abbrv_re, entity_re


def contains_entity(text, entity_set, abbrv_set):
    ent_list = list()
    ents = entity_set.findall(text)
    if ents:
        ent_list.extend(ents)
    abbrvs = abbrv_set.findall(text)
    if abbrvs:
        for a in abbrvs:
            ent_list.append(a)
    return ent_list


def count_glob(src_path, glob, entities, abbrvs):
    start = time.time()
    entity_count = defaultdict(int)
    doc_entity = dict()
    for sent_dict, fname in cu.raw2dict(src_path, glob):
        for sd in sent_dict:
            sent = sd["sentence"]
            ent_list = contains_entity(sent, entities, abbrvs)
            for ent in ent_list:
                entity_count[ent.strip()] += 1
        doc_entity[fname] = sorted(
            entity_count.items(), key=lambda x: x[1], reverse=True
        )
        entity_count = defaultdict(int)
    logger.info("time : {:}".format(cu.format_time(time.time() - start)))
    return doc_entity


def entity_mentions(entity_file, corpus_dir, glob):
    abbvs, ents = get_entity(entity_file)
    return count_glob(corpus_dir, glob, ents, abbvs)


if __name__ == "__main__":
    import json
    import os
    from argparse import ArgumentParser
    import gamechangerml.src.text_classif.utils.log_init as li

    li.initialize_logger(to_file=False, log_name="none")

    fp = os.path.split(__file__)
    fp = "python " + fp[-1]
    parser = ArgumentParser(
        prog=fp,
        description="outputs counts of entity mentions in each document",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        dest="input_path",
        type=str,
        help="corpus path",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--entity-file",
        dest="entity_file",
        type=str,
        help="list of entities with their abbreviations",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-json",
        dest="output_json",
        type=str,
        required=True,
        help="output path for .csv files",
    )
    parser.add_argument(
        "-g",
        "--glob",
        dest="glob",
        type=str,
        required=True,
        help="file pattern to match",
    )
    args = parser.parse_args()

    output = entity_mentions(args.entity_file, args.input_path, args.glob)
    output = json.dumps(output)
    with open(args.output_json, "w") as f:
        f.write(output)

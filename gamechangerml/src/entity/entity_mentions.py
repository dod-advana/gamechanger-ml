"""
usage: python entity_mentions.py [-h] -i INPUT_PATH -e ENTITY_FILE -o
                                 OUTPUT_JSON -g GLOB -t {mentions,spans}

brute force extraction of entity mentions in documents

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH
                        corpus path
  -e ENTITY_FILE, --entity-file ENTITY_FILE
                        csv of entities, abbreviations, and entity type
  -o OUTPUT_JSON, --output-json OUTPUT_JSON
                        output path for .csv files
  -g GLOB, --glob GLOB  file pattern to match
  -t {mentions,spans}, --task {mentions,spans}
                        what do you want to run?
"""
import json
import logging
import os
import re
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import gamechangerml.src.text_classif.utils.classifier_utils as cu

logger = logging.getLogger(__name__)

ETYPE = "etype"
SENT = "sentence"
TEXTTYPE = "raw_text"
ABBRV = "ABBRV"
ENTITY = "entity"


def entity_csv_to_df(entity_csv):
    if not os.path.isfile(entity_csv):
        raise FileNotFoundError("can't find {}".format(entity_csv))
    df = pd.read_csv(entity_csv, names=[ENTITY, ETYPE])
    df = df.replace(np.nan, "")
    return df


def make_entity_re(entity_csv):
    """
    Creates regular expressions for long form entities and abbreviations.
    These are large alternations. No magic. A mapping of entity to entity
    type (`entity2type`) is created for subsequent lookup in creating
    NER training data.

    Args:
        entity_csv (str): csv with entries consisting of
            *ENT*, *ETYPE*

    Returns:
        SRE_Pattern, SRE_Pattern, Dict
    """
    df = entity_csv_to_df(entity_csv)
    entity2type = dict()
    long_forms = list()
    short_forms = list()

    for _, row in df.iterrows():
        entity, etype = row[ENTITY], row[ETYPE]
        entity2type[entity.lower()] = etype
        if ABBRV in etype:
            short_forms.append(entity)
        else:
            long_forms.append(entity)

    long_forms.sort(key=lambda s: len(s), reverse=True)
    short_forms.sort(key=lambda s: len(s), reverse=True)

    entity_re = "|".join(
        ["(\\b" + re.escape(e.strip()) + "\\b)" for e in long_forms]
    )
    entity_re = re.compile(entity_re, re.I)

    abbrv_re = "|".join(
        (["(\\b" + re.escape(a.strip()) + "\\b)" for a in short_forms])
    )
    abbrv_re = re.compile(abbrv_re)

    return abbrv_re, entity_re, entity2type


def contains_abbrv(text, abbrv_re):
    return [a.group() for a in abbrv_re.finditer(text)]


def contains_non_abbrv(text, entity_re):
    return [e.group() for e in entity_re.finditer(text)]


def contains_entity(text, entity_re, abbrv_re):
    """
    Finds all the entities in the text, returning a list with every
    instance of the entity. If no entities are found, an empty list is
    returned.

    Args:
        text (str): text to search
        entity_re (SRE_Pattern): compiled regular expression for entities
        abbrv_re (SRE_Pattern): compiled regular expression for abbreviations

    Returns:
        List
    """

    entity_list = contains_abbrv(text,  abbrv_re)
    entity_list.extend(contains_non_abbrv(text, entity_re))

    return entity_list


def unnest_nested_abbrv_entity(entity_span_list, abbrv_span_list):
    """
    If an abbreviation entity is part of a larger entity, exclude it as
    an abbreviation entity.

    Args:
        entity_span_list: list of entities and their spans
            List[(entity_text, (start_position, end_position))]

        abbrv_span_list: list of abbreviations and their spans
            List[(abbreviation_text, (start_position, end_position))]

    Returns:
        List[tuple(str, tuple(int, int))]

    """
    contained = list()
    unnested_ents = list()

    for ent, ent_spans in entity_span_list:
        unnested_ents.append((ent, ent_spans))
        for abbrv, abbrv_spans in abbrv_span_list:
            if (
                abbrv_spans[0] >= ent_spans[0]
                and abbrv_spans[1] <= ent_spans[1]  # not needed now, but...
            ):
                contained.append((abbrv, ent))  # good for testing, for now
                logger.debug("'{}' contained in '{}'".format(abbrv, ent))
            else:
                unnested_ents.append((abbrv, abbrv_spans))
    if contained:
        logger.debug(set(contained))
    return unnested_ents


def entities_spans_in_text(text, entity_re, abbrv_re):
    """
    Finds all the entities in the text, returning a list with every
    instance of the entity. If no entities are found, an empty list is
    returned.

    Args:
        text (str): text to search

        entity_re (SRE_Pattern): compiled regular expression

        abbrv_re (SRE_Pattern): compiled regular expression

    Returns:
        List[tuple(str, tuple(int, int))]
    """
    ent_span_list = list()
    abbrv_span_list = list()

    # TODO use list comprehension
    for mobj in entity_re.finditer(text):
        entity_span = (mobj.group(), (mobj.start(), mobj.end()))
        ent_span_list.append(entity_span)

    for mobj in abbrv_re.finditer(text):
        entity_span = (mobj.group(), (mobj.start(), mobj.end()))
        abbrv_span_list.append(entity_span)

    if ent_span_list:
        unnested_entity = unnest_nested_abbrv_entity(
            ent_span_list, abbrv_span_list
        )
        logger.debug(unnested_entity)
        logger.debug("-------")
        return unnested_entity
    else:
        ent_span_list.extend(abbrv_span_list)
        logger.debug(ent_span_list)
        logger.debug("-------")
        return ent_span_list


def count_entity_mentions(corpus_dir, glob, entity_re, abbrv_re):
    """
    For each matching document, list each entity and its frequency of
    occurrence.

    Args:
        corpus_dir (str): directory containing the corpus
        glob (str): file matching glob
        entity_re (SRE_Pattern): compiled regular expression for long forms
        abbrv_re (SRE_Pattern): compiled regular expression for short forms

    Returns:
        Dict[List[tuple]] : key is the document name, each tuple is
            (entity, frequency)
    """
    nfiles = cu.nfiles_in_glob(corpus_dir, glob)
    entity_count = defaultdict(int)
    doc_entity = dict()

    r2d = cu.raw2dict(corpus_dir, glob)
    for sent_list, fname in tqdm(r2d, total=nfiles, desc="docs"):
        for sd in sent_list:
            sent = sd[SENT]
            ent_list = contains_entity(sent, entity_re, abbrv_re)
            for ent in ent_list:
                entity_count[ent] += 1
        doc_entity[fname] = sorted(
            entity_count.items(), key=lambda x: x[1], reverse=True
        )
        entity_count = defaultdict(int)
    return doc_entity


def count_entity_mentions_in_corpus(entity_file, corpus_dir, glob):
    """
    Convenience method for `count_entity_mentions()`.

    Args:
        entity_file (str): entity / abbreviation files
        corpus_dir (str): corpus directory
        glob (str): file matching expression

    Returns:
        Dict[List[tuple]] : key is the document name, each tuple is
            (entity, frequency)
    """
    abbrvs, ents, _ = make_entity_re(entity_file)
    return count_entity_mentions(corpus_dir, glob, ents, abbrvs)


def entities_in_raw(entity_file, corpus_dir, glob):
    """
    Finds each occurrence of an entity along with its span.

    Args:
        entity_file (str): entity / abbreviation files
        corpus_dir (str): corpus directory
        glob (str): file matching

    Returns:
       str, List[tuple, tuple]
    """
    abbrv_re, entity_re, _ = make_entity_re(entity_file)
    for fname, json_doc in cu.gen_gc_docs(corpus_dir, glob):
        text = cu.scrubber(json_doc[TEXTTYPE])
        entity_spans = entities_spans_in_text(text, entity_re, abbrv_re)
        yield fname, entity_spans


def entities_and_spans_by_doc(entity_file, corpus_dir, glob):
    """
    Wrapper for `entities_from_raw()`
    Args:
        entity_file (str): entity / abbreviation files
        corpus_dir (str): corpus directory
        glob (str): file matching

    Returns:
        Dict[str: tuple[str, tuple(int, int)]
    """
    nfiles = cu.nfiles_in_glob(corpus_dir, glob)
    entity_span_d = dict()
    efr = entities_in_raw(entity_file, corpus_dir, glob)
    for fname, entity_spans in tqdm(efr, total=nfiles):
        entity_span_d[fname] = entity_spans
    return entity_span_d


def entity_types_in_text(text, entity_re, abbrv_re, entity2type):
    entity_list = contains_entity(text, entity_re, abbrv_re)
    entity_types = [
        entity2type[entity.lower()] for entity in entity_list if entity
    ]
    return entity_types, entity_list


def entity_types_in_sentences(sent_list, entity_re, abbrv_re, entity2type):
    entity_count = Counter()
    names = list()
    unique_labels = sorted(list(set(list(entity2type.values()))))
    names.extend(unique_labels)
    names.append(SENT)
    df = pd.DataFrame(columns=names)

    for text in tqdm(sent_list, desc="sentences"):
        row = {etype: 0 for etype in unique_labels}
        row[SENT] = text
        ent_types, ent_list = entity_types_in_text(
            text, entity_re, abbrv_re, entity2type
        )
        entity_count.update(ent_list)
        if not ent_types:
            continue
        unique_in_sent = list(set(ent_types))
        for ent_type in unique_in_sent:
            row[ent_type] = 1
        df = df.append(row, ignore_index=True)
    return df, entity_count


def sentences_with_entities(sent_list, entity_re, abbrv_re):
    ent_df = pd.DataFrame()
    no_ent_df = pd.DataFrame()
    for _, row in tqdm(sent_list.iterrows(), total=len(ent_df)):
        if contains_entity(row[SENT], entity_re, abbrv_re):
            ent_df = ent_df.append(row, ignore_index=True)
        else:
            no_ent_df = no_ent_df.append(row, ignore_index=True)
    return ent_df, no_ent_df


def _json_out(output_dict, output_path):
    if output_dict:
        enc_output = json.dumps(output_dict)
        with open(output_path, "w") as fp:
            fp.write(enc_output)
        logger.info("output written to : {}".format(args.output_json))
    else:
        logger.warning("no output produced")


if __name__ == "__main__":
    from argparse import ArgumentParser

    import gamechangerml.src.text_classif.utils.log_init as li

    li.initialize_logger(to_file=False, log_name="none")

    parser = ArgumentParser(
        prog="python " + os.path.split(__file__)[-1],
        description="brute force extraction of entity mentions in documents",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        dest="input_path",
        type=str,
        required=False,
        help="corpus path",
    )
    parser.add_argument(
        "-e",
        "--entity-file",
        dest="entity_file",
        type=str,
        required=True,
        help="csv of entities, abbreviations, and entity type",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        dest="output_path",
        type=str,
        required=False,
        help="output path for .csv files",
    )
    parser.add_argument(
        "-g",
        "--glob",
        dest="glob",
        type=str,
        required=False,
        help="file pattern to match",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        choices=["mentions", "spans", "profile", "sentences"],
        required=True,
        help="what do you want to run?",
    )
    parser.add_argument(
        "-s",
        "--sentence_csv",
        dest="sentence_csv",
        required=False,
        default=None,
        help=".csv of sentences",
    )

    args = parser.parse_args()
    if not os.path.isfile(args.entity_file):
        raise ValueError("cannot find {}".format(args.entity_file))
    logger.info(args.sentence_csv)

    output = None
    start = time.time()

    if args.task == "spans":
        output = entities_and_spans_by_doc(
            args.entity_file, args.input_path, args.glob
        )
        _json_out(output, args.output_path)
    elif args.task == "mentions":
        output = count_entity_mentions_in_corpus(
            args.entity_file, args.input_path, args.glob
        )
        _json_out(output, args.output_path)
    elif args.task == "profile":
        df_ = pd.read_csv(args.sentence_csv, names=["fname", "label", SENT])
        sent_list_ = df_[SENT].to_list()
        abbrv_re_, entity_re_, entity2type_ = make_entity_re(args.entity_file)
        out_df, ent_counts = entity_types_in_sentences(
            sent_list_, entity_re_, abbrv_re_, entity2type_
        )
        out_df.to_csv("corpus_entity_profile.csv", header=True, index=False)
        for k, v in ent_counts.most_common():
            logger.info("{:>6,d} : {}".format(v, k))
    elif args.task == "sentences":
        abbrv_re_, entity_re_, _ = make_entity_re(args.entity_file)
        df_in = pd.read_csv(args.sentence_csv, names=["fname", "label", SENT])
        out_ent_df, out_no_ent_df = sentences_with_entities(df_in, entity_re_, abbrv_re_)
        out_ent_df.to_csv(args.output_path, header=False, index=False)
        out_no_ent_df.to_csv("all_dod_dim_no_entity_sentences.csv", header=True, index=False)

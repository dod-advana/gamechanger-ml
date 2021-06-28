"""
usage: python predict_table.py [-h] -m MODEL_PATH -d DATA_PATH [-b BATCH_SIZE]
                               [-l MAX_SEQ_LEN] -g GLOB [-o OUTPUT_CSV] -a
                               AGENCIES_PATH [-r]

Binary classification of each sentence in the files matching the 'glob' in
data_path

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model-path MODEL_PATH
                        directory of the torch model
  -d DATA_PATH, --data-path DATA_PATH
                        path holding the .json corpus files
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size for the data samples; default=8
  -l MAX_SEQ_LEN, --max-seq-len MAX_SEQ_LEN
                        maximum sequence length, 128 to 512; default=128
  -g GLOB, --glob GLOB  file glob pattern
  -o OUTPUT_CSV, --output-csv OUTPUT_CSV
                        the .csv for output
  -a AGENCIES_PATH, --agencies-path AGENCIES_PATH
                        the .csv for agency abbreviations
  -r, --raw-output      write the results of the classifier / entity
                        attachment
"""
import logging
import os
import time
import pandas as pd

from datetime import date

import gamechangerml.src.text_classif.utils.classifier_utils as cu
from gamechangerml.src.featurization.abbreviations_utils import (
    get_references,
    get_agencies_dict,
    get_agencies,
)
from gamechangerml.src.text_classif.utils.entity_coref import EntityCoref
from gamechangerml.src.text_classif.utils.log_init import initialize_logger
from gamechangerml.src.text_classif.utils.resp_stats import count_output

logger = logging.getLogger(__name__)


def _agg_stats(df, model_name='NA', seq_len='NA', batch_size='NA'):
    run_date = date.today().strftime("%d%m%Y")
    resp_per_doc, resp_no_entity, n_uniq_entities, n_docs = count_output(df)
    if resp_per_doc:
        df_resp_doc = pd.DataFrame(
            list(resp_per_doc.items()), columns=["doc", "count"]
        )
        df_resp_doc.to_csv("resp-in-doc-stats_{md}_{sl}_{bs}_{rd}.csv".format(md=model_name, sl=seq_len, bs=batch_size, rd=run_date), index=False)
    if resp_no_entity:
        df_resp_no_e = pd.DataFrame(
            list(resp_per_doc.items()), columns=["doc", "count"]
        )
        df_resp_no_e.to_csv("resp-no-entity-stats_{md}_{sl}_{bs}_{rd}.csv".format(md=model_name, sl=seq_len, bs=batch_size, rd=run_date), index=False)


def predict_table(
    model_path, data_path, glob, max_seq_len, batch_size, output_csv, stats_path
):
    """
    See the preamble (help) for a description of these arguments.

    For each file matching `glob`, the `raw_text` is parsed into sentences
    and run through the classifier. Recognized entities are then associated
    with sentences classified as `1` or `responsibility`. The final output
    is assembled by using sentences classified as `1` with organization
    information, references, document title, etc.

    Returns:
        pandas.DataFrame
    """
    if not os.path.isdir(data_path):
        raise ValueError("no data path {}".format(data_path))
    if not os.path.isdir(model_path):
        raise ValueError("no model path {}".format(model_path))

    rename_dict = {
        "entity": "Organization / Personnel",
        "sentence": "Responsibility Text",
        "agencies": "Other Organization(s) / Personnel Mentioned",
        "refs": "Documents Referenced",
        "title": "Document Title",
        "source": "Source Document",
    }

    start = time.time()
    entity_coref = EntityCoref()
    entity_coref.make_table(
        model_path,
        data_path,
        glob,
        max_seq_len,
        batch_size,
    )
    df = entity_coref.to_df()
    df = df[df.top_class == 1].reset_index()

    logger.info("retrieving agencies csv")
    duplicates, aliases = get_agencies_dict(args.agencies_path)
    df["agencies"] = get_agencies(
        file_dataframe=df[['Responsibility Text']],
        doc_dups=None,
        duplicates=duplicates,
        agencies_dict=aliases,
    )

    df["refs"] = get_references(df, doc_title_col="src")
    renamed_df = df.rename(columns=rename_dict)

    final_df = renamed_df[
        [
            "Source Document",
            "Document Title",
            "Organization / Personnel",
            "Responsibility Text",
            "Other Organization(s) / Personnel Mentioned",
            "Documents Referenced",
            "Org Filter"
        ]
    ]
    if output_csv is not None:
        final_df.to_csv(output_csv, index=False)
        logger.info("final csv written")
    if stats_path is not False:
        model = model_path.split('/')[-1]
        _agg_stats(final_df, model, max_seq_len, batch_size)
    elapsed = time.time() - start

    logger.info("total time : {:}".format(cu.format_time(elapsed)))
    return final_df


if __name__ == "__main__":
    from argparse import ArgumentParser

    desc = "Binary classification of each sentence in the files "
    desc += "matching the 'glob' in data_path"
    fp = os.path.split(__file__)
    fp = "python " + fp[-1]
    parser = ArgumentParser(prog=fp, description=desc)
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        type=str,
        required=True,
        help="directory of the pytorch model",
    )
    parser.add_argument(
        "-d",
        "--data-path",
        dest="data_path",
        type=str,
        required=True,
        help="path holding the .json corpus files",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        help="batch size for the data samples; default=8",
    )
    parser.add_argument(
        "-l",
        "--max-seq-len",
        dest="max_seq_len",
        type=int,
        default=128,
        help="maximum sequence length, 128 to 512; default=128",
    )
    parser.add_argument(
        "-g",
        "--glob",
        dest="glob",
        type=str,
        required=True,
        help="file glob pattern",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        dest="output_csv",
        type=str,
        default=None,
        help="the .csv for output",
    )
    parser.add_argument(
        "-a",
        "--agencies-path",
        dest="agencies_path",
        type=str,
        required=True,
        help="the .csv for agency abbreviations",
    )
    parser.add_argument(
        "-s",
        "--stats-path",
        action="store_false",
        dest="stats_path",
        help="write aggregate statistics output to file",
    )

    initialize_logger(to_file=False, log_name="none")

    args = parser.parse_args()

    _ = predict_table(
        args.model_path,
        args.data_path,
        args.glob,
        args.max_seq_len,
        args.batch_size,
        args.output_csv,
        args.stats_path,
    )

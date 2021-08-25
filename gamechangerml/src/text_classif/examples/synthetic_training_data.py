"""
test/utils script for potentially creating synthetic responsibility classifier training data
"""

import pandas as pd
import logging
import spacy

# resp_verbs = ['shall', 'approves', 'issues', 'assists', 'establishes']
resp_verbs = ["shall"]

spacy_model_ = spacy.load("en_core_web_lg")

logger = logging.getLogger(__name__)


def create_synth_statement(org_list, verb_list, resp_list):
    output_list = []
    for i in org_list:
        for j in verb_list:
            for k in resp_list:
                output_list.append("The " + i + " " + j + " " + k)
    return output_list


if __name__ == "__main__":
    from argparse import ArgumentParser

    desc = "testing script for creating synthetic training data"
    parser = ArgumentParser(prog="python table.py", description=desc)

    parser.add_argument(
        "-i", "--input-dir", dest="input_dir", type=str, required=True
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=str, required=True
    )
    parser.add_argument("-r", "--orgs-file", dest="orgs", type=str, default="")
    parser.add_argument(
        "-t", "--training-data", dest="training_data", type=str, default=""
    )

    args = parser.parse_args()

    orgs_file = pd.read_csv(args.orgs, sep="/t", header=None)
    training_data = pd.read_csv(args.training_data)

    orgs = []
    for i in orgs_file[0]:
        orgs.append(i.split(" (")[0])

    resp_frags = []
    for i in training_data["text"]:
        resp_frags.append(i.split("shall ")[1])

    sample = create_synth_statement(
        orgs, resp_verbs, resp_frags[110000:110010]
    )
    for i in sample:
        logger.info(i)

    logger.info("synthetic data created")

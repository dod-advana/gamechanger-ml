import argparse
import logging
import os
from datetime import datetime

from gamechangerml.configs.config import D2VConfig, DefaultConfig
from gamechangerml.src.search.query_expansion.build_ann_cli import (
    build_qe_model as bqe,
)
from gamechangerml.src.utilities import utils
from gamechangerml.src.search.sent_transformer.model import SentenceEncoder
from gamechangerml import REPO_PATH

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
"""
Usage:
    to use on CLI:
        # to save with a test flag  {YYYYMMDD-test}
        python -m dataSciece.scripts.run_train_models --nameflag test --save-remote True --corpus PATH_TO_CORPUS

        # no flag
        python -m dataSciece.scripts.run_train_models --save-remote True --corpus PATH_TO_CORPUS

    NOTE: MUST RUN setup_env.sh to save to S3
optional arguments:
    -h, help messages
    -c, --corpus CORPUS DIRECTORY
    -s, --saveremote BOOLEAN SAVE TO S3 (TRUE/FALSE)
    -f, --nameflag STR tag for the model name
    -d, --modeldest MODEL PATH DIR
    -v  , --validate True or False
    -x  , --experimentName name of experiment, if exists will add as a run
"""

modelname = datetime.now().strftime("%Y%m%d")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "--nameflag",
        "-f",
        dest="nameflag",
        type=str,
        help="model name flag i.e. best",
    )
    parser.add_argument("--saveremote", "-s", dest="save",
                        help="save to s3 flag")
    parser.add_argument(
        "--modeldest", "-d", dest="model_dest", help="model destination dir"
    )
    parser.add_argument("--corpus", "-c", dest="corpus", help="corpus dir")
    parser.add_argument(
        "--validate",
        "-v",
        dest="validate",
        help="flag for running validation tests and appending metrics",
    )
    parser.add_argument(
        "--experimentName",
        "-x",
        dest="experimentName",
        default=modelname,
        help="experiement name, keep consistent if you want to compare in mlfow",
    )
    parser.add_argument(
        "--sentenceTrans",
        "-s",
        dest="sentenceTrans",
        default=False,
        help="True or False Flag for building sentence index",
    )
    parser.add_argument(
        "--gpu",
        "-gpu",
        dest="usegpu",
        default=False,
        help="True or False Flag for using gpu",
    )
    args = parser.parse_args()
    if args.nameflag:
        modelname = f"{modelname}-{args.nameflag}"
    if args.save == "True" or args.save == "true":
        save = True
    else:
        save = False
    if args.validate:
        validate = True
    else:
        validate = False
    model_dest = args.model_dest
    if args.sentenceTrans == "True" or args.sentenceTrans == "true":
        sentTrans = True
    else:
        sentTrans = False
    if args.usegpu == "True" or args.usegpu == "true":
        gpu = True
    else:
        gpu = False

    if not model_dest:
        model_dest = DefaultConfig.LOCAL_MODEL_DIR
    run_train(
        model_id=modelname,
        save_remote=save,
        corpus_dir=args.corpus,
        model_dest=model_dest,
        exp_name=args.experimentName,
        validate=validate,
        sentenceTrans=sentTrans,
        gpu=gpu,
    )

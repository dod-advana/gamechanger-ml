"""Utility functions for scheduling ml builds based on events

Also see gamechangerml.src.services.s3_service.py
"""

from threading import current_thread
from os import makedirs
from os.path import join, exists, basename
import datetime
from gamechangerml.src.services.s3_service import S3Service
from gamechangerml.src.utilities import configure_logger
from gamechangerml.configs import S3Config
from gamechangerml.api.utils import processmanager
from gamechangerml.api.fastapi.routers.controls import (
    download_corpus,
    train_model,
    train_qexp,
)
from gamechangerml.api.utils.threaddriver import MlThread
import os
from queue import Queue
from gamechangerml.src.data_transfer import download_corpus_s3

from fastapi import APIRouter, Response

from gamechangerml.api.fastapi.settings import (
    CORPUS_DIR,
    S3_CORPUS_PATH,
    CORPUS_EVENT_TRIGGER_VAL,
)


async def corpus_update_event(
    s3_corpus_dir: str,
    corpus_dir: str = "gamechangerml/corpus",
    bucket=None,
    logger=None,
) -> bool:
    if logger is None:
        logger = configure_logger()

    if bucket is None:
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)

    process = processmanager.ml_change_event

    try:
        logger.info("ML EVENT - Checking corpus staleness")

        s3_filter = bucket.objects.filter(Prefix=f"{s3_corpus_dir}/")
        total = len(list(s3_filter))
        local_corpus_size = len(os.listdir(corpus_dir))

        ratio = local_corpus_size / total
        if ratio < CORPUS_EVENT_TRIGGER_VAL:
            logger.info("ML EVENT - Corpus is stale - downloading data")
            # trigger a thread to update corpus and build selected models
            logger.info("Attempting to download corpus from S3")

            thread_args = {
                "args": {
                    "logger": logger,
                    "s3_args": {
                        "s3_corpus_dir": s3_corpus_dir,
                        "output_dir": CORPUS_DIR,
                        "logger": logger,
                    },
                    "qexp_model_dict": {
                        "build_type": "qexp",
                        "upload": True,
                        "version": datetime.datetime.today().strftime("%Y%m%d"),
                    },
                }
            }

            logger.info(thread_args)
            ml_event_thread = MlThread(run_update, thread_args)
            ml_event_thread.start()
            processmanager.running_threads[ml_event_thread.ident] = ml_event_thread
            processmanager.update_status(
                processmanager.ml_change_event, 0, 1, thread_id=ml_event_thread.ident
            )

    except Exception:
        logger.exception("Failed to update corpus or train models")
        processmanager.update_status(
            process, failed=True, thread_id=current_thread().ident
        )


def run_update(args):
    logger = args["logger"]
    logger.info("Attempting to download corpus from S3")
    download_corpus_s3(**args["s3_args"])
    logger.info("Attempting to build Qexp")
    model_dict = args["qexp_model_dict"]
    train_qexp(model_dict)
    processmanager.update_status(
        processmanager.ml_change_event,
        1,
        1,
        thread_id=current_thread().ident,
    )

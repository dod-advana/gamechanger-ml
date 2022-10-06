"""Utility functions for scheduling ml builds based on events

Also see gamechangerml.src.services.s3_service.py
"""

from threading import current_thread
from os import makedirs
from os.path import join, exists, basename
from datetime import datetime, timezone
from gamechangerml.src.services.s3_service import S3Service
from gamechangerml.src.utilities import configure_logger
from gamechangerml.configs import S3Config
from gamechangerml.api.utils import processmanager
from gamechangerml.api.fastapi.routers.controls import (
    train_qexp,
    train_sentence,
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
    latest_intel_model_encoder,
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
        last_mod_list = []
        if os.path.isdir(corpus_dir):
            local_corpus_size = len(os.listdir(corpus_dir))
            if local_corpus_size > 0:
                local_corpus_last_updated = datetime.fromtimestamp(
                    os.stat(corpus_dir).st_mtime
                ).astimezone(timezone.utc)
                for obj in s3_filter:
                    last_mod_list.append(obj.last_modified)

                last_mod_list = [
                    dates
                    for dates in last_mod_list
                    if dates > local_corpus_last_updated
                ]
                ratio = len(last_mod_list) / local_corpus_size
            else:
                ratio = 1
        else:
            ratio = 1
        if ratio > CORPUS_EVENT_TRIGGER_VAL:
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
                        "version": datetime.today().strftime("%Y%m%d"),
                    },
                    "sent_model_dict": {
                        "build_type": "sentence",
                        "upload": True,
                        "version": datetime.today().strftime("%Y%m%d"),
                        "encoder_model": str(latest_intel_model_encoder.value).split(
                            "/"
                        )[-1],
                        "gpu": True,
                    },
                }
            }

            logger.info(thread_args)
            ml_event_thread = MlThread(run_update, thread_args)
            ml_event_thread.start()
            processmanager.running_threads[ml_event_thread.ident] = ml_event_thread
            processmanager.update_status(
                processmanager.ml_change_event, 0, 3, thread_id=ml_event_thread.ident
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
    processmanager.update_status(
        processmanager.ml_change_event,
        1,
        3,
        thread_id=current_thread().ident,
    )
    logger.info("Attempting to build Qexp")
    model_dict = args["qexp_model_dict"]
    train_qexp(model_dict)
    processmanager.update_status(
        processmanager.ml_change_event,
        2,
        3,
        thread_id=current_thread().ident,
    )
    logger.info("Attempting to build Sentence Index")
    model_dict = args["sent_model_dict"]

    train_sentence(model_dict)
    processmanager.update_status(
        processmanager.ml_change_event,
        3,
        3,
        thread_id=current_thread().ident,
    )

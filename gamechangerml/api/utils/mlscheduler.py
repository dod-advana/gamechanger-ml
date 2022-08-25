"""Utility functions for scheduling ml builds based on events

Also see gamechangerml.src.services.s3_service.py
"""

from threading import current_thread
from os import makedirs
from os.path import join, exists, basename

from gamechangerml.src.services.s3_service import S3Service
from gamechangerml.src.utilities import configure_logger
from gamechangerml.configs import S3Config
from gamechangerml.api.utils import processmanager
from gamechangerml.api.fastapi.routers.controls import download_corpus
from gamechangerml.api.utils.threaddriver import MlThread

from gamechangerml.src.data_transfer import delete_local_corpus, download_corpus_s3

import os
from queue import Queue

from fastapi import APIRouter, Response

from gamechangerml.api.fastapi.settings import (
    CORPUS_DIR,
    S3_CORPUS_PATH,
)


def check_corpus_diff(
    s3_corpus_dir: str,
    corpus_dir: str = "gamechangerml/corpus",
    bucket=None,
    logger=None,
) -> bool:
    if logger is None:
        logger = configure_logger()

    if bucket is None:
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)

    process = processmanager.corpus_download

    try:
        logger.info("Getting ratio")

        s3_filter = bucket.objects.filter(Prefix=f"{s3_corpus_dir}/")
        total = len(list(s3_filter))
        local_corpus_size = len(os.listdir(corpus_dir))
        logger.info(f"local corpus size {local_corpus_size}")
        logger.info(f"total corpus size {total}")

        ratio = local_corpus_size / total
        logger.info(ratio)
        if ratio < 0.95:
            logger.info("Corpus is out of date - downloading data")
            try:
                logger.info("Attempting to download corpus from S3")
                # grabs the s3 path to the corpus from the post in "corpus"
                # then passes in where to dowload the corpus locally.

                args = {
                    "s3_corpus_dir": s3_corpus_dir,
                    "output_dir": CORPUS_DIR,
                    "logger": logger,
                }

                logger.info(args)
                corpus_thread = MlThread(download_corpus_s3, args)
                corpus_thread.start()
                processmanager.running_threads[corpus_thread.ident] = corpus_thread
                processmanager.update_status(
                    processmanager.corpus_download, 0, 1, thread_id=corpus_thread.ident
                )
            except Exception as e:
                logger.exception("Could not get corpus from S3")
                processmanager.update_status(
                    processmanager.corpus_download,
                    failed=True,
                    message=e,
                    thread_id=corpus_thread.ident,
                )
    except Exception:
        logger.exception("Failed to read corpus in S3")
        processmanager.update_status(
            process, failed=True, thread_id=current_thread().ident
        )

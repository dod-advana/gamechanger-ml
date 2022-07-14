"""Utility functions for downloading objects from S3. 

Also see gamechangerml.src.services.s3_service.py
"""

from threading import current_thread
from os.path import join, exists, basename

from gamechangerml.src.services.s3_service import S3Service
from gamechangerml.utils import configure_logger
from gamechangerml.configs import S3Config
from gamechangerml.api.utils import processmanager
from gamechangerml.src.data_transfer import delete_local_corpus

def download_corpus_s3(
    s3_corpus_dir,
    output_dir="corpus",
    bucket=None,
    logger=None,
    rm_existing=True,
):
    """Download the corpus from S3.

    Args:
        s3_corpus_dir (str): Path to S3 directory that contains the corpus.
        output_dir (str, optional): Path to directory to download files to.
            Defaults to "corpus".
        bucket (boto3.resources.factory.s3.Bucket or None, optional): Bucket to
            download from. If None, uses S3Service.connect_to_bucket(). Default
            is None.
        logger (logging.Logger or None, optional): If None, uses
            configure_logger(). Default is None.
        rm_existing (bool, optional): True to delete existing files in the
            output directory before downloading, False otherwise. Default is
            True.

    Returns:
        list of str: Paths (in S3) to downloaded files.
    """
    if logger is None:
        logger = configure_logger()

    if bucket is None:
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)

    if rm_existing:
        success = delete_local_corpus(output_dir, logger)
        if not success:
            return []

    corpus = []
    process = processmanager.corpus_download

    try:
        filter = bucket.objects.filter(Prefix=f"{s3_corpus_dir}/")
        total = len(list(filter))
        num_completed = 0

        # Initialize Progress
        processmanager.update_status(
            process, num_completed, total, thread_id=current_thread().ident
        )

        logger.info("Downloading corpus from " + s3_corpus_dir)
        for obj in filter:
            corpus.append(obj.key)
            filename = basename(obj.key)
            local_path = join(output_dir, filename)
            # Only grab file if it is not already downloaded
            if ".json" in filename and not exists(local_path):
                bucket.Object(obj.key).download_file(local_path)
                num_completed += 1
            # Update Progress
            processmanager.update_status(
                process,
                num_completed,
                total,
                thread_id=current_thread().ident,
            )
    except Exception:
        logger.exception("Failed to download corpus from S3.")
        processmanager.update_status(
            process, failed=True, thread_id=current_thread().ident
        )

    return corpus

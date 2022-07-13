"""Utility functions for downloading objects from S3."""

from os.path import join
from gamechangerml.src.services.s3_service import S3Service
from gamechangerml.utils import configure_logger
from gamechangerml.configs import S3Config


def get_model_s3(
    s3_model_dir, filename, download_dir="", bucket=None, logger=None
):
    """Download a model from S3.

    Args:
        s3_model_dir (str): Path to S3 directory which contains the model to 
            download.
        filename (str): File name of the model to download.
        download_dir (str, optional): Path to local directory to put downloaded
            files. Defaults to "".
        bucket (boto3.resources.factory.s3.Bucket or None, optional): Bucket to
            upload to. If None, uses S3Service.connect_to_bucket(). Default is 
            None.
        logger (logging.Logger or None, optional): If None, uses 
            configure_logger(). Default is None.

    Returns:
        list of str: Paths to locally downloaded files.
    """
    if logger is None:
        logger = configure_logger()

    if bucket is None:
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)

    files = []
    model_path = join(s3_model_dir, filename)

    try:
        for obj in bucket.objects.filter(Prefix=model_path):
            if obj.size != 0:
                logger.info(f"Downloading {obj.key}")
                bucket.download_file(
                    obj.key, join(download_dir, obj.key.split("/")[-1])
                )
                files.append(join(download_dir, obj.key.split("/")[-1]))
    except Exception:
        logger.exception(f"Failed to download S3 model at {model_path}.")

    return files

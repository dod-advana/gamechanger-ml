import boto3
import os
import logging

bucket_name = os.getenv("AWS_BUCKET_NAME", default="advana-data-zone")
env = os.getenv("ENV_TYPE")
logger = logging.getLogger("gamechanger")

def s3_connect():
    conn = boto3.Session()
    s3 = conn.resource("s3")
    bucket = s3.Bucket(bucket_name)
    return bucket


def upload_file(filepath, s3_fullpath):
    """upload_file - uploads files to s3 bucket
    Args:
        filepath: path to file
        s3_fullpath: exact path you want to save it

    Returns:
    """
    bucket = s3_connect()
    try:
        bucket.upload_file(filepath, s3_fullpath)
    except Exception as e:
        logger.debug(f"could not upload {filepath} to {s3_fullpath}")
        logger.debug(e)

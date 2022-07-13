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

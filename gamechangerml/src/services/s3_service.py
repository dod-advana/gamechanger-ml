from boto3 import Session


class S3Service:
    """This class is responsible for providing connection to S3, including 
    uploading and downloading files.
    """

    @staticmethod
    def connect_to_bucket(bucket_name, logger):
        """Connect to S3 bucket.

        Returns:
            boto3.resources.factory.s3.Bucket or None: The Bucket if 
                connection was successful. Otherwise, None.
        """
        try:
            session = Session()
            s3 = session.resource("s3")
            bucket = s3.Bucket(bucket_name)
        except Exception:
            logger.exception("Failed to connect to S3 bucket.")
            bucket = None

        return bucket

    @staticmethod
    def upload_file(bucket, filepath, s3_fullpath, logger):
        """Upload a file to the S3 bucket.
        
        Args:
            bucket (boto3.resources.factory.s3.Bucket): Bucket to upload to. 
                See S3Service.connect_to_bucket().
            filepath (str): Path to the file to upload.
            s3_fullpath (str): Path to save to in S3.
            logger (logging.Logger)

        Returns:
            None
        """
        try:
            bucket.upload_file(filepath, s3_fullpath)
        except Exception:
            logger.exception(
                f"Failed to upload file at {filepath} to S3 {s3_fullpath}."
            )

    @staticmethod
    def put_object(bucket, object, s3_dir, file_name, logger):
        """_summary_

        Args:
            bucket (boto3.resources.factory.s3.Bucket): Bucket to upload to. 
                See S3Service.connect_to_bucket().
            object (TODO): TODO
            s3_dir (str): Directory path to store the object in S3.
            file_name (str): File name to give the object in S3.
            logger (logging.Logger)

        Returns:
            None
        """
        try:
            bucket.put_object(Body=object, Key=s3_dir + file_name)
        except Exception:
            logger.exception(f"Failed to put {file_name} in S3.")

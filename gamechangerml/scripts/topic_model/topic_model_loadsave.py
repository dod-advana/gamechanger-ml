from gamechangerml.src.utilities.utils import get_models_list
from gamechangerml.src.services import S3Service
from gamechangerml.data_transfer import download_model_s3
from gamechangerml import REPO_PATH
import os
import sys

topic_model_dir = os.path.join(
    REPO_PATH, "gamechangerml/models/topic_models/models/"
)

os.chdir(topic_model_dir)
s3_models_dir = "models/topic_models/"

try:
    sys.argv[1]
except:
    raise Exception(
        '\nArgument not specified. Specify "load" or "save" as an argument into the shell script.'
    )

bucket = S3Service.connect_to_bucket()
if bucket is None:
    raise Exception("Failed to connect to S3.")

# if we're loading models from s3
if sys.argv[1].lower() == "load":
    print("\nLoading models from s3 \n")

    # download everything from s3
    print(get_models_list(s3_models_dir))
    for s in get_models_list(s3_models_dir):
        download_model_s3(filename=s[0], s3_model_dir=s3_models_dir)
    print("\nFinished")

# if we're saving models into s3
elif sys.argv[1].lower() == "save":
    print("\nSaving models into s3\n")

    # check if the directory is empty
    print(f"List of files being uploaded: {os.listdir()}")
    if not os.listdir():
        raise Exception(
            "\nModels directory is empty. Load models into the directory before saving to s3."
        )

    # upload everything in the directory to s3
    for s in os.listdir():
        print(f"Uploading {s} ...")
        S3Service.upload_file(
            bucket=bucket, filename=s, s3_fullpath=s3_models_dir + s
        )
    print("\nFinished")
else:
    raise Exception(
        'Specify "load" or "save" as an argument into the shell script.'
    )

import logging
from os import rename, makedirs
from os.path import join, isdir, basename
import glob
import tarfile
import typing as t
from pathlib import Path
from gamechangerml.src.services import S3Service
from gamechangerml.configs import S3Config
from gamechangerml import REPO_PATH

logger = logging.getLogger("gamechanger")


def create_model_schema(model_dir, file_prefix):
    num = 0
    while isdir(join(model_dir, file_prefix)):
        file_prefix = f"{file_prefix.split('_')[0]}_{num}"
        count += 1
    
    dirpath = join(model_dir, file_prefix)
    makedirs(dirpath)

    logger.info(f"Created directory: {dirpath}.")


def get_transformers(model_path="transformers_v4/transformers.tar", overwrite=False, bucket=None):
    if bucket is None:
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)

    models_path = join(REPO_PATH, "gamechangerml/models")
    try:
        if glob.glob(join(models_path, "transformer*")):
            if not overwrite:
                print(
                    "transformers exists -- not pulling from s3, specify overwrite = True"
                )
                return
        for obj in bucket.objects.filter(Prefix=model_path):
            print(obj)
            bucket.download_file(
                obj.key, join(models_path, obj.key.split("/")[-1])
            )
            compressed = obj.key.split("/")[-1]
        cache_path = join(models_path, compressed)
        print("uncompressing: " + cache_path)
        compressed_filename = compressed.split(".tar")[0]
        if isdir(f"{models_path}/{compressed_filename}"):
            rename(
                f"{models_path}/{compressed_filename}",
                f"{models_path}/{compressed_filename}_backup",
            )
        tar = tarfile.open(cache_path)
        tar.extractall(models_path)
        tar.close()
    except Exception:
        print("cannot get transformer model")
        raise


def get_sentence_index(model_path="sent_index/", overwrite=False, bucket=None):
    if bucket is None:
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)

    models_path = join(REPO_PATH, "gamechangerml/models")
    try:
        if glob.glob(join(models_path, "sent_index*")):
            if not overwrite:
                print(
                    "sent_index exists -- not pulling from s3, specify overwrite = True"
                )
                return
        for obj in bucket.objects.filter(Prefix=model_path):
            print(obj)
            bucket.download_file(
                obj.key, join(models_path, obj.key.split("/")[-1])
            )
            compressed = obj.key.split("/")[-1]
        cache_path = join(models_path, compressed)
        print("uncompressing: " + cache_path)
        compressed_filename = compressed.split(".tar")[0]
        if isdir(f"{models_path}/{compressed_filename}"):
            rename(
                f"{models_path}/{compressed_filename}",
                f"{models_path}/{compressed_filename}_backup",
            )
        tar = tarfile.open(cache_path)
        tar.extractall(models_path)
        tar.close()
    except Exception:
        print("cannot get transformer model")
        raise


def view_all_datasets(bucket=None):
    if bucket is None:
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)

    prefix = "eval_data/"
    all_datasets = set()
    for obj in bucket.objects.filter(Prefix=prefix):
        object_key = obj.key.replace(prefix, "")
        object_key = object_key.split("/")[:2]
        object_key = "/".join(object_key)
        all_datasets.add(object_key)

    logger.info("Available datasets:")
    for dataset in all_datasets:
        logger.info(f"\t{dataset}")


def download_eval_data(dataset_name, save_dir, version=None, bucket=None):
    """
    store_eval_data - download eval data to local directory
        params: folder_path (str), folder containing data
                version (int), version number of dataset
        output:
    """
    save_dir = join(save_dir, dataset_name)
    if not isdir(save_dir):
        mkdir(save_dir)

    if bucket is None:
        bucket = S3Service.connect_to_bucket(S3Config.BUCKET_NAME, logger)

    prefix = "eval_data/"
    try:
        all_datasets = set()
        for obj in bucket.objects.filter(Prefix=prefix):
            object_key = obj.key.replace(prefix, "")
            dataset = object_key.split("/")[0]
            all_datasets.add(dataset)
    except:
        logger.debug(
            "Failed to query dataset version. Maybe the dataset doesn't exist")

    if dataset_name not in all_datasets:
        logger.debug(f"{dataset_name} not in available datasets.")
        logger.debug(f"Available datasets are {list(all_datasets)}")
        return None

    prefix = f"eval_data/{dataset_name}/"
    try:
        all_versions = set()
        for obj in bucket.objects.filter(Prefix=prefix):
            object_key = obj.key.replace(prefix, "")
            object_ver = int(object_key.split("/")[0][1:])
            all_versions.add(object_ver)
    except:
        logger.debug(
            "Failed to query dataset version. Maybe the dataset doesn't exist")

    if version is None:
        version = max(all_versions)
    elif version not in all_versions:
        logger.debug(f"Version {version} not found.")
        logger.debug(f"Available versions are {list(all_versions)}")
        return None

    logger.info(f"Downloading {dataset_name} version {version}...")
    prefix += f"v{version}"
    try:
        for obj in bucket.objects.filter(Prefix=prefix):
            fname = obj.key.split("/")[-1]
            save_name = join(save_dir, fname)
            bucket.download_file(obj.key, save_name)
    except:
        logger.debug(f"Failed to download {obj.key}")


def create_tgz_from_dir(
    src_dir: t.Union[str, Path],
    dst_archive: t.Union[str, Path],
    exclude_junk: bool = False,
) -> None:
    with tarfile.open(dst_archive, "w:gz") as tar:
        tar.add(src_dir, arcname=basename(src_dir))

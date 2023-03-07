from os.path import join, getctime, isdir, exists
from os import listdir, rmdir, remove, makedirs
from shutil import rmtree
from json import load, loads, dump
from logging import Logger
from typing import Union, List
from .numpy_utils import NumpyJSONEncoder


def open_txt(filepath):
    """Opens a txt file"""
    with open(filepath, "r") as fp:
        return fp.readlines()


def save_json(filename, path, data):
    """Saved a json file"""
    filepath = join(path, filename)
    with open(filepath, "w") as outfile:
        return dump(data, outfile, cls=NumpyJSONEncoder)


def open_json(filename, path=""):
    """Opens a json file"""
    with open(join(path, filename), "r") as f:
        return load(f)


def open_jsonl(filename, path):
    """Opens a jsonl file"""
    with open(join(path, filename), "r") as json_file:
        json_list = list(json_file)

    data = []
    for json_str in json_list:
        result = loads(json_str)
        data.append(result)

    return data


def delete_files(path, logger=None):
    """Deletes all files in a directory"""
    use_logger = isinstance(logger, Logger)

    info_msg = f"Cleaning up: removing test files from {str(path)}"
    if use_logger:
        logger.info(info_msg)
    else:
        print(info_msg)

    for file in listdir(path):
        fpath = join(path, file)
        print(fpath)
        try:
            rmtree(fpath)
        except OSError:
            remove(fpath)
    try:
        rmdir(path)
    except OSError as e:
        error_msg = "Error: %s : %s" % (path, e.strerror)
        if use_logger:
            logger.error(error_msg)
        else:
            print(error_msg)


def get_most_recently_changed_dir(parent_dir, logger=None):
    use_logger = isinstance(logger, Logger)

    subdirs = [
        join(parent_dir, d)
        for d in listdir(parent_dir)
        if isdir(join(parent_dir, d))
    ]
    if len(subdirs) > 0:
        return max(subdirs, key=getctime)
    else:
        error_msg = f"There are no subdirectories to retrieve most recent data from within {parent_dir}."
        if use_logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None


def create_directory_if_not_exists(directory, logger=None):
    """Checks if a directory exists, if it does not makes the directory"""
    use_logger = isinstance(logger, Logger)

    if not exists(directory):
        info_msg = "Creating new directory {}".format(directory)
        if use_logger:
            logger.info(info_msg)
        else:
            print(info_msg)
        makedirs(directory)

    return directory


def get_json_paths_for_directory(
    directory_path, strict_file_names: Union[List[str], None] = None
):
    """Get paths to JSON files for the given directory.

    Args:
        directory_path (str): Directory path.
        strict_file_names (Union[List[str], None], optional): If None, returns 
            paths for all JSON files in the directory. Otherwise, only returns
            paths for the file names in this list. Defaults to None.
    Returns:
        list of str: JSON file paths
    """
    file_names = [fn for fn in listdir(directory_path) if fn[-5:] == ".json"]

    if strict_file_names:
        file_names = [fn for fn in file_names if fn in strict_file_names]

    return [join(directory_path, fn) for fn in file_names]

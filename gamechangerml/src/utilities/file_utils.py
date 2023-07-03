from os.path import join, getctime, isdir, exists
from os import listdir, rmdir, remove, makedirs
from shutil import rmtree
from json import load, loads, dump
from logging import Logger
import pickle
from .numpy_utils import NumpyJSONEncoder


def save_pickle(object, path, name=None):
    if name:
        path = join(path, name)

    with open(path, "wb") as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


def open_txt(filepath):
    """Opens a txt file"""
    with open(filepath, "r") as fp:
        return fp.readlines()


def save_json(filename, path, data):
    """Saved a json file"""
    filepath = join(path, filename)
    with open(filepath, "w") as outfile:
        return dump(data, outfile, cls=NumpyJSONEncoder)


def open_json(filename, path):
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

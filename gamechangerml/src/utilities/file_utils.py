from os.path import join
from json import load, loads, dump
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


def open_json(filename, path):
    """Opens a json file"""
    with open(join(path, filename)) as f:
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

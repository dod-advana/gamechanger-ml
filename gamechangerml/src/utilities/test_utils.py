import os
import json
from datetime import date
from gamechangerml.api.utils.logger import logger

# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182
class TimeoutException(Exception):   # Custom exception class
    pass

# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182
def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

def check_file_size(filename, path):
    return os.path.getsize(os.path.join(path, filename))

# from create_embeddings.py
def get_user(logger):
    try:
        user = os.environ.get("GC_USER", default="root")
        if (user =="root"):
            user = str(os.getlogin())
    except Exception as e:
        user = "unknown"
        logger.info("Could not get system user")
        logger.info(e)

def save_json(filename, path, data):

    filepath = os.path.join(path, filename)
    with open(filepath, "w") as outfile: 
        return json.dump(data, outfile)

def open_json(filename, path):
    with open(os.path.join(path, filename)) as f:
        return json.load(f)

def open_jsonl(filename, path):

    with open(os.path.join(path, filename), 'r') as json_file:
        json_list = list(json_file)

    data = []
    for json_str in json_list:
        result = json.loads(json_str)
        data.append(result)
    
    return data

def open_txt(filepath):
    with open(filepath, "r") as fp:
        return fp.readlines()

def timestamp_filename(filename, extension):
    today = date.today()
    formatted = '_'.join([filename, today.strftime("%Y-%m-%d")])
    return formatted + extension

def check_directory(directory):

    if not os.path.exists(directory):
        logger.info("Creating new directory {}".format(directory))
        os.makedirs(directory)

    return directory
import os
import string
import re
import json
from datetime import date
from gamechangerml.api.utils.logger import logger

# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182
class TimeoutException(Exception):   # Custom exception class
    pass

# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182
def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

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

# Source: https://rajpurkar.github.io/SQuAD-explorer/
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()
import os
import string
import re
import json
from datetime import date
from gamechangerml.api.utils.logger import logger
import torch

# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182
class TimeoutException(Exception):   # Custom exception class
    pass

# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182
def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

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

# from sentence_transformers==2.0.0
#https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py
def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
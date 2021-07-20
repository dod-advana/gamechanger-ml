import os
import string
import re
import json

def open_json(filename, path):
    with open(os.path.join(filename, path)) as f:
        return json.load(f)

def open_txt(filepath):
    with open(filepath, "r") as fp:
        return fp.readlines()

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
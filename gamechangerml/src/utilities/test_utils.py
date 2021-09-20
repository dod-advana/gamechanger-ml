import math
import os
import json
from datetime import date
import signal
from gamechangerml.api.utils.logger import logger

# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182
class TimeoutException(Exception):   # Custom exception class
    pass

def init_timer():
    '''Creates a timer using signal'''
    # https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182
    def timeout_handler(signum, frame):   # Custom signal handler
        raise TimeoutException
    signal.signal(signal.SIGALRM, timeout_handler)
    logger.info("Created timer.")

    return 

def check_file_size(filename, path):
    '''Returns the filesize (in bytes) of a file'''
    return os.path.getsize(os.path.join(path, filename))

# from create_embeddings.py
def get_user(logger):
    '''Gets user or sets value to 'unknown' (from create_embeddings.py)'''
    try:
        user = os.environ.get("GC_USER", default="root")
        if (user =="root"):
            user = str(os.getlogin())
    except Exception as e:
        user = "unknown"
        logger.info("Could not get system user")
        logger.info(e)

def save_json(filename, path, data):
    '''Saved a json file'''
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as outfile: 
        return json.dump(data, outfile)

def open_json(filename, path):
    '''Opens a json file'''
    with open(os.path.join(path, filename)) as f:
        return json.load(f)

def open_jsonl(filename, path):
    '''Opens a jsonl file'''
    with open(os.path.join(path, filename), 'r') as json_file:
        json_list = list(json_file)

    data = []
    for json_str in json_list:
        result = json.loads(json_str)
        data.append(result)
    
    return data

def open_txt(filepath):
    '''Opens a txt file'''
    with open(filepath, "r") as fp:
        return fp.readlines()

def timestamp_filename(filename, extension):
    '''Makes a filename that include a %Y-%m-%d timestamp'''
    today = date.today()
    formatted = '_'.join([filename, today.strftime("%Y%m%d")])
    return formatted + extension

def check_directory(directory):
    '''Checks if a directory exists, if it does not makes the directory'''
    if not os.path.exists(directory):
        logger.info("Creating new directory {}".format(directory))
        os.makedirs(directory)

    return directory

def get_most_recent_eval(directory):
    '''Gets the most recent eval json from a directory'''
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    evals = [f for f in files if f.split('.')[-1]=='json']
    if evals:
        evals.sort(key=lambda x:int(x.split('_')[-1].split('.')[0].replace('-', '')))
        return evals[-1]
    else:
        return ''

def collect_evals(directory):
    '''Checks if a model directory has any evaluations'''
    sub_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    eval_dirs = [os.path.join(directory, d) for d in sub_dirs if d.split('_')[0]=='evals']
    if not eval_dirs:
        return {}
    else:
        evaldict = {}
        for i in eval_dirs:
            name = i.split('_')[1]
            file = get_most_recent_eval(i)
            if file != '':
                evaldict[name] = open_json(file, i)
            else:
                evaldict[name] = {}
        return evaldict

def clean_nans(value):
    '''Replaces null value with 0'''
    if value == None or math.isnan(value):
        return 0
    else:
        return value
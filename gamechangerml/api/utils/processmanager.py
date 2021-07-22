import threading
from datetime import datetime
from gamechangerml.api.utils.redisdriver import CacheVariable

# Process Keys
corpus_download = "corpus_download"
loading_corpus = "loading_corpus"
training = "model_training"

# the dictionary that holds all the progress values
PROCESS_STATUS = CacheVariable("process_status", True)
COMPLETED_PROCESS = CacheVariable("completed_process", True)
thread_lock = threading.Lock()

default_flags = {
    corpus_download: False,
    training: False,
    loading_corpus:False
}

PROCESS_STATUS.value = {
    "flags":default_flags
}
COMPLETED_PROCESS.value = []


def update_status(key, progress = 0, total = 100, failed = False):
    # log update at most 1000 times
    update_step = total/1000
    if update_step < 1:
        update_step = 1
        
    if progress == total or failed:
        date = datetime.now()
        date_string = date.strftime("%Y-%m-%d %H:%M:%S")
        completed = {
            "process": key,
            "total": total,
            "date": date_string
        }
        with thread_lock:
            if key in PROCESS_STATUS.value:
                temp = PROCESS_STATUS.value
                temp.pop(key, None)
                temp["flags"][key] = False
                PROCESS_STATUS.value = temp
            if not failed:
                completed_list = COMPLETED_PROCESS.value
                completed_list.append(completed)
                COMPLETED_PROCESS.value = completed_list
    elif progress%update_step ==0:
        status = {
            "progress":progress,
            "total":total
        }
        with thread_lock:
            status_dict = PROCESS_STATUS.value
            status_dict[key] = status
            status_dict["flags"][key] = False
            PROCESS_STATUS.value = status_dict

def set_flags(key, value):
    with thread_lock:
            status_dict = PROCESS_STATUS.values
            status_dict["flags"][key] = value
            PROCESS_STATUS.value = status_dict
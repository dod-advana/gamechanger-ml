import threading
from datetime import datetime
from gamechangerml.api.utils.redisdriver import CacheVariable

# Process Keys
clear_corpus = "corpus: corpus_download"
corpus_download = "corpus: corpus_download"
delete_corpus = "corpus: delete_corpus"
loading_corpus = "training: load_corpus"
training = "training: train_model"
reloading = "models: reloading_models"

# the dictionary that holds all the progress values
try:
    PROCESS_STATUS = CacheVariable("process_status", True)
    COMPLETED_PROCESS = CacheVariable("completed_process", True)
    thread_lock = threading.Lock()
    default_flags = {
        corpus_download: False,
        clear_corpus: False,
        training: False,
        loading_corpus:False,
        reloading:False
    }

    PROCESS_STATUS.value = {
        "flags":default_flags
    }
    COMPLETED_PROCESS.value = []
except Exception as e:
    print(e)


def update_status(key, progress = 0, total = 100, failed = False):

    try:
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
        else:
            status = {
                "progress":progress,
                "total":total
            }
            with thread_lock:
                status_dict = PROCESS_STATUS.value
                status_dict[key] = status
                status_dict["flags"][key] = True
                PROCESS_STATUS.value = status_dict
    except Exception as e:
        print(e)

def set_flags(key, value):
    with thread_lock:
            status_dict = PROCESS_STATUS.values
            status_dict["flags"][key] = value
            PROCESS_STATUS.value = status_dict

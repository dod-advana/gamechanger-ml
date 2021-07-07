import threading
from datetime import datetime
from gamechangerml.api.utils.redisdriver import CacheVariable

# Process Keys
corpus_download = "corpus_download"
training_progress = "training_progress"

# the dictionary that holds all the progress values
PROCESS_STATUS = CacheVariable("process_status", True)
COMPLETED_PROCESS = CacheVariable("completed_process", True)
thread_lock = threading.Lock()

PROCESS_STATUS.value = {}
COMPLETED_PROCESS.value = []

def update_status(key, progress, total):
    if progress == total:
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
                PROCESS_STATUS.value = temp
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
            PROCESS_STATUS.value = status_dict
import threading
from datetime import datetime
from gamechangerml.api.utils.redisdriver import CacheVariable

# Process Keys
corpus_download = "corpus_download"
training_progress = "training_progress"

# the dictionary that holds all the progress values
PROCESS_STATUS = CacheVariable("process_status", True)
COMPLETED_PROCESS = CacheVariable("completed_process", list = True)
thread_lock = threading.Lock()

if PROCESS_STATUS.value == None:
    PROCESS_STATUS.value = {}

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
            PROCESS_STATUS.value[key] = None
            COMPLETED_PROCESS.value.push(completed)
    else:
        status = {
            "progress":progress,
            "total":total
        }
        with thread_lock:
            PROCESS_STATUS.value[key] = status
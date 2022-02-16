import threading
from datetime import datetime
from gamechangerml.api.utils.redisdriver import CacheVariable
from gamechangerml.api.fastapi.settings import logger
# Process Keys
clear_corpus = "corpus: corpus_download"
corpus_download = "corpus: corpus_download"
delete_corpus = "corpus: delete_corpus"
s3_file_download = "s3: file_download"
s3_dependency = "s3: dependency_download"
loading_corpus = "training: load_corpus"
loading_data = "training: load_data"
training = "training: train_model"
reloading = "models: reloading_models"
ltr_creation = "training: ltr_creation"
topics_creation = "models: topics_creation"

running_threads = {}

# the dictionary that holds all the progress values
try:
    PROCESS_STATUS = CacheVariable("process_status", True)
    COMPLETED_PROCESS = CacheVariable("completed_process", True)
    thread_lock = threading.Lock()
    default_flags = {
        corpus_download: False,
        clear_corpus: False,
        training: False,
        loading_corpus: False,
        reloading: False,
        ltr_creation: False,
        topics_creation: False,
        s3_file_download: False,
        s3_dependency: False,
        loading_data: False

    }

except Exception as e:
    print(e)

if PROCESS_STATUS.value == None:
    PROCESS_STATUS.value = {"flags": default_flags}
if COMPLETED_PROCESS.value == None:
    COMPLETED_PROCESS.value = []


def update_status(key, progress=0, total=100, message="", failed=False, thread_id="", completed_max=20):

    try:
        if progress == total or failed:
            date = datetime.now()
            date_string = date.strftime("%Y-%m-%d %H:%M:%S")
            completed = {
                "process": key,
                "total": total,
                "message": message,
                "date": date_string,
            }
            with thread_lock:
                if key in PROCESS_STATUS.value:
                    temp = PROCESS_STATUS.value
                    tempProcess = temp.pop(key, None)
                    if key in temp["flags"]:
                        temp["flags"][key] = False
                    PROCESS_STATUS.value = temp
                    if tempProcess['thread_id'] in running_threads:
                        del running_threads[tempProcess['thread_id']]
                if not failed:
                    completed_list = COMPLETED_PROCESS.value
                    if len(completed_list) == completed_max:
                        completed_list.pop(0)
                    completed_list.append(completed)
                    COMPLETED_PROCESS.value = completed_list
                else:
                    completed['date'] = 'Failed'
                    completed_list = COMPLETED_PROCESS.value
                    completed_list.append(completed)
                    COMPLETED_PROCESS.value = completed_list
        else:
            status = {"progress": progress, "total": total}
            with thread_lock:
                status_dict = PROCESS_STATUS.value

                if key not in status_dict:
                    status['thread_id'] = thread_id
                    status_dict[key] = status
                else:
                    status_dict[key].update(status)

                if key in status_dict["flags"]:
                    status_dict["flags"][key] = True
                PROCESS_STATUS.value = status_dict
    except Exception as e:
        print(e)


def set_flags(key, value):
    with thread_lock:
        status_dict = PROCESS_STATUS.values
        status_dict["flags"][key] = value
        PROCESS_STATUS.value = status_dict

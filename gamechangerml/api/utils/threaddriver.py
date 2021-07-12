import threading
import json
from gamechangerml.api.utils.logger import logger
# A class that takes in a function and a dictionary of arguments.
# The keys in args have to match the parameters in the function.
class MlThread(threading.Thread):
    def __init__(self, function, args):
        super(MlThread, self).__init__()
        self.function = function
        self.args = args
    def run(self):
        try:
            self.function(**self.args)
        except Exception as e:
            logger.error(e)
            logger.info("Thread errored out attempting " + self.function.__name__ + " with parameters: " + json.dumps(self.args))


# Pass in a function and args which is an array of dicts
# A way to load mulitple jobs and run them on threads.
# join is set to false unless we need to collect the results immediately.
def run_threads(function_list, args_list = [], join = False):
    threads = []
    for i, function in enumerate(function_list):
        args = {}
        if i < len(args_list):
            args = args_list[i]
        thread = MlThread(function, args)  
        threads.append(thread)
        thread.start()
    # If we join the threads the function will wait until they have all completed.
    if join:
        for thread in threads:
            thread.join()
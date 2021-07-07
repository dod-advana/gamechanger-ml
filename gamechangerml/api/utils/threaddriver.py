import threading
# A class that takes in a function and a dictionary of arguments.
# The keys in args have to match the parameters in the function.
class MlThread(threading.Thread):
    def __init__(self, function, args):
        super(MlThread, self).__init__()
        self.function = function
        self.args = args
    def run(self):
        self.function(**self.args)

# Pass in a function and args which is an array of dicts
# A way to load mulitple jobs and run them on threads.
# join is set to false unless we need to collect the results immediately.
def run_threads(function, args, join = False):
    threads = []
    for arg in args:
        thread = MlThread(function, arg)  
        threads.append(thread)
        thread.start()
    # If we join the threads the function will wait until they have all completed.
    if join:
        for thread in threads:
            thread.join()
import threading
# 
class MlThread(threading.Thread):
    def __init__(self, function, args):
        super(MlThread, self).__init__()
        self.function = function
        self.args = args
    def run(self):
        self.function(**self.args)

#
def run_threads(function, args):
    threads = []
    for arg in args:
        thread = MlThread(function, arg)  
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
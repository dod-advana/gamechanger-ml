import threading

def run_threads(function, args):
    threads = []
    for arg in args:
        thread = threading.Thread(target=function, args=arg)  
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
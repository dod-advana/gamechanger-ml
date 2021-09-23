import sys

workers = 1
loglevel = 'debug'

def worker_int(worker):
    print('Exit because of worker failure')
    sys.exit(1)
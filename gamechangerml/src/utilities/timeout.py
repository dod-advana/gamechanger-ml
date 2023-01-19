import signal

# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time/25027182


class TimeoutException(Exception):  # Custom exception class
    pass


def init_timer(logger):
    """Creates a timer using signal"""

    def timeout_handler(signum, frame):  # Custom signal handler
        raise TimeoutException

    signal.signal(signal.SIGALRM, timeout_handler)
    logger.info("Created timer.")

    return

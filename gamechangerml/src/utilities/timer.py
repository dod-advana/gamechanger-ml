import time
import logging

logger = logging.getLogger(__name__)


class Timer:
    def __enter__(self, level="debug"):
        """
        A simple timer class. Usage is

            `with Timer():
                < code to be timed >'

        If `level` is not None, it will generate a log message with  the elapsed
        time according specified `level` - either logger.DEBUG or logger.INFO. 
        If `level` is not None and is anything other than "debug", the level is 
        logger.INFO
        If `level` is None, no message is logged.

        Args:
            level (str or None): If None, no message will be logged. Otherwise, 
                anything other than None or the default "debug" will log as 
                logging.INFO; default is "debug".

        """
        self._level = level
        self.t0 = time.time()

    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0
        if self._level is None:
            pass
        elif self._level == "debug":
            logger.debug(
                "elapsed time: {:0.3f}s".format(self.elapsed)
            )
        else:
            logger.info("elapsed time: {:0.3f}s".format(time.time() - self.t0))

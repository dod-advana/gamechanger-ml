import logging
from logging import handlers
import sys
from gamechangerml.src.utilities import configure_logger

# set loggers
logger = configure_logger()
glogger = logging.getLogger("gunicorn.error")

try:
    # glogger.addHandler(ch)
    log_file_path = "gamechangerml/api/logs/gc_ml_logs.txt"
    fh = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=2000000, backupCount=1, mode="a"
    )
    logger.info(f"ML API is logging to {log_file_path}")

    # fh.setFormatter(log_formatter)
    logger.addHandler(fh)
    glogger.addHandler(fh)
except Exception as e:
    logger.warn(e)

import sys
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager


def get_timestamp(hms_seperator=':'):
    return time.strftime(f"%Y-%m-%d %H{hms_seperator}%M{hms_seperator}%S", time.localtime())


def exception_traceback(e):
    e_tb = "\n".join(traceback.format_tb(e.__traceback__) + [str(e)])
    return e_tb


@contextmanager
def try_except_pass():
    try:
        yield
    except:
        pass


def setup_logger(log_file_name=None, name=None):
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s | %(levelname)5s | %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)5s | %(message)s'))
    logger.addHandler(handler)
    if log_file_name is not None:
        handler = RotatingFileHandler(log_file_name, maxBytes=5 * 1024 * 1024, delay=True)
        logger.addHandler(handler)
        logger.handlers[-1].setFormatter(logger.handlers[0].formatter)
    return logger

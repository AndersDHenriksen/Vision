import time
import traceback
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

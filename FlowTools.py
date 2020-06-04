import time
import traceback
import functools

def get_timestamp(hms_seperator=':'):
    return time.strftime(f"%Y-%m-%d %H{hms_seperator}%M{hms_seperator}%S", time.localtime())


def exception_traceback(e):
    e_tb = "\n".join(traceback.format_tb(e.__traceback__) + [str(e)])
    return e_tb


def try_except_pass(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            value = func(*args, **kwargs)
        except:
            value = None
        return value
    return wrapper_decorator

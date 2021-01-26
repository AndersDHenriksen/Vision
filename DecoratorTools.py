import functools
from pathlib import Path
import cv2


def try_except_pass(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            value = func(*args, **kwargs)
        except:
            value = None
        return value
    return wrapper_decorator


def loop_until_true(func):
    @functools.wraps(func)
    def wrapper_decorator(operator, *args, **kwargs):
        return_value = False
        while not return_value:
            return_value = func(operator, *args, **kwargs)
        return return_value
    return wrapper_decorator


def profile(func):
    from line_profiler import LineProfiler  # conda install line_profiler

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            prof.print_stats()

    return wrapper


def path2image(enforce_grayscale=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Do something before
            if isinstance(args[0], str) or isinstance(args[0], Path):
                image = cv2.imread(str(args[0]), cv2.IMREAD_GRAYSCALE if enforce_grayscale else None)
                args = (image, *args[1:])
            return func(*args, **kwargs)
        return wrapper
    return decorator

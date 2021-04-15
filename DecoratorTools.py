import sys
import hashlib
import functools
from pathlib import Path
import numpy as np
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


def md5(input_string):
    return hashlib.md5(str(input_string).encode('utf8')).hexdigest()


def path2image(enforce_grayscale=False, load_from_shared_memory=False):
    from multiprocessing import shared_memory
    from PIL import Image

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(args[0], str) or isinstance(args[0], Path):
                if load_from_shared_memory:
                    try:
                        existing_shm = shared_memory.SharedMemory(name=md5(args[0]))
                        img_shape = Image.open(args[0]).size
                        image = np.ndarray(img_shape[::-1], dtype=np.uint8, buffer=existing_shm.buf)
                    except FileNotFoundError:
                        print("Image could not be found in shared memory. Loading from disc.")
                        image = cv2.imread(str(args[0]), cv2.IMREAD_GRAYSCALE if enforce_grayscale else -1)
                else:
                    image = cv2.imread(str(args[0]), cv2.IMREAD_GRAYSCALE if enforce_grayscale else -1)
                args = (image, *args[1:])
            return func(*args, **kwargs)
        return wrapper
    return decorator


def put_images_into_shared_memory(img_dir, pattern="*.png", enforce_grayscale=False):
    from multiprocessing import shared_memory
    shms = []
    for img_path in Path(img_dir).rglob(pattern):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE if enforce_grayscale else -1)
        shm = shared_memory.SharedMemory(name=md5(img_path), create=True, size=img.nbytes)
        shms.append(shm)
        img_shm = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
        img_shm[:] = img[:]
        print(f"Loaded {md5(img_path)}: {img_path}")
    input("Images stored in shared memory. Input enter to shutdown shared memory.")
    for shm in shms:
        shm.unlink()


if __name__ == "__main__":
    print(f"DecoratorTools called with argv: {sys.argv[1:]}")
    put_images_into_shared_memory(sys.argv[1], enforce_grayscale=True)

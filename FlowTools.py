#! C:\Conda\python.exe
from pathlib import Path
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import configparser
import subprocess


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


@contextmanager
def timeit():
    start = time.time()
    yield
    print(f"Call took: {time.time()-start:.3f} sec")


def setup_logger(log_file_name=None, name='vision', n_log_files=100):
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s | %(levelname)5s | %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)5s | %(message)s'))
    logger.addHandler(handler)
    if log_file_name is not None:
        handler = RotatingFileHandler(log_file_name, maxBytes=5 * 1024 * 1024, backupCount=n_log_files)
        logger.addHandler(handler)
        logger.handlers[-1].setFormatter(logger.handlers[0].formatter)
    return logger


VERSION_PATH = Path('project_config.ini')


def bump_version():
    config = configparser.ConfigParser()
    config.read(VERSION_PATH)
    if 'VERSION' in config:
        version_integers = config['VERSION']['code-version'].split('.')
        version_integers[2] = str(int(version_integers[2]) + 1)
        config['VERSION']['code-version'] = '.'.join(version_integers)
    else:
        config['VERSION'] = {'code-version': '0.0.0'}
    config['VERSION']['push-time'] = get_timestamp()
    config['VERSION']['remote.origin'] = subprocess.run('git config --get remote.origin.url',
                                                        capture_output=True).stdout.decode('utf-8').strip()
    with open(VERSION_PATH, 'w') as configfile:
        config.write(configfile)


def get_code_version():
    if not VERSION_PATH.exists():
        bump_version()
    config = configparser.ConfigParser()
    config.read(VERSION_PATH)
    return config['VERSION']['code-version']


if __name__ == "__main__":  # For easy git pre-push hook
    bump_version()

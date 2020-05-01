import traceback


def exception_traceback(e):
    e_tb = "\n".join(traceback.format_tb(e.__traceback__) + [str(e)])
    return e_tb
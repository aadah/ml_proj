import sys


def my_print(obj, same_line=False):
    if same_line:
        print obj,
    else:
        print obj

    sys.stdout.flush()

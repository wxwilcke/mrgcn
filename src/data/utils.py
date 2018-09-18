#!/usr/bin/python3

import logging
from os import access, F_OK, R_OK, W_OK
from os.path import split


logger = logging.getLogger(__name__)

def is_readable(filename):
    path = split(filename)[0]
    if not access(path, F_OK):
        raise OSError(":: Path does not exist: {}".format(path))
    elif not access(path, R_OK):
        raise OSError(":: Path not readable by user: {}".format(path))

    return True

def is_writable(filename):
    path = split(filename)[0]
    if not access(path, F_OK):
        raise OSError(":: Path does not exist: {}".format(path))
    elif not access(path, W_OK):
        raise OSError(":: Path not writeable by user: {}".format(path))

    return True

def is_gzip(filename):
    return True if filename.endswith('.gz') else False

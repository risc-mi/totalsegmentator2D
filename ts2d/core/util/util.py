import os
import shutil
from typing import Any, Union, List

import numpy as np

from ts2d.core.util.log import warn
from ts2d.core.util.types import as_tuple, as_list, as_set


def parse_int(val: Any, err=None):
    """
    Converts a value to int, if not possible, returns err
    :param val: string to parse an int from
    :param err: value to return if parsing fails
    :return: int value
    """
    try:
        if isinstance(val, int):
            return val
        else:
            return int(str(val).strip())
    except:
        return err


def parse_float(val: str, err=np.nan):
    """
    Converts a string value to float, if not possible, returns err
    :param val: string to parse a float from
    :param err: value to return if parsing fails
    :return: float value
    """
    try:
        return float(str(val).strip())
    except:
        return err


def short_message(msg, limit: int=100):
    """
    reduces the message to printable characters and restricts the length
    :param msg: original message
    :param limit: character limit
    :return: shortened message
    """
    res = ''.join(c for c in str(msg) if c.isprintable())
    if len(res) > limit:
        res = '...{}'.format(res[-limit:])
    return res


def mkdirs(*args):
    """
    Creates one or many directories in case they don't exist.
    param *args: arbitrary number of entries, each a path or list of path to the directories to create
    Examples:
        makedirs('./a')
        makedirs('./a', './b')
        makedirs(['./a', './b')
        makedirs(['./a', './b'], './c)
    """
    for arg in args:
        if isinstance(arg, str):
            arg = [arg]
        for p in arg:
            os.makedirs(p, exist_ok=True)

def rmdirs(dps: Union[str, list]):
    """
    removes a directory and its contents
    :param dps: path or collection of paths to the directories
    """
    for dp in as_list(dps):
        if os.path.exists(dp):
            try:
                shutil.rmtree(dp)
            except Exception as ex:
                warn("Failed to remove directory: {}".format(ex))


def format_array(a, p=3, sep=', '):
    """
    converts an array or scalar to a string
    :param a: array to convert
    :param p: precision to use
    :param sep: separator to use between values
    """
    a = as_tuple(a)
    return sep.join(np.format_float_positional(v, precision=p, unique=False) for v in a)

def removeall(fns: Union[List[str], str]):
    """
    removes one or many files
    :param fns: one or many paths to remove
    """
    fns = as_set(fns)
    for fn in fns:
        if os.path.isfile(fn):
            os.remove(fn)

def isemptydir(path: str):
    """
    returns True if the specified path refers to an empty directory
    """
    from glob import glob
    if os.path.exists(path) and os.path.isdir(path):
        return not list(fn for fn in glob(os.path.join(path, '**'), recursive=True) if os.path.isfile(fn))
    return False

def removeprefix(a: str, b: str):
    """removes a prefix b from the left side of a if it exists"""
    return a[len(b):] if a.startswith(b) else a

def unit_vector(v, axis=-1, div0=np.nan):
    """
    converts a vector or vector matrix to convert to unit vectors
    :param v: vector or vector matrix to convert
    :param axis: axis to normalize, defaults to the last axis
    :param div0: value to return where the unit length is zero
    :return: normalized unit vector result
    """
    v = np.asanyarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis)
    reorder = axis != 0
    if reorder:
        v = np.moveaxis(v, axis, 0)
    v = np.divide(v, n, out=np.full_like(v, div0), where=n!=0)
    if reorder:
        v = np.moveaxis(v, 0, axis)
    return v

def is_nnunet_multilabel():
    try:
        from nnunetv2 import multilabel
        return multilabel
    except ModuleNotFoundError:
        raise RuntimeError("nnunetv2 is not installed")
    except ImportError:
        return False
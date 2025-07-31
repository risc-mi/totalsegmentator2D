from typing import Optional, Union

import numpy as np

from ts2d.core.util.util import format_array
from ts2d.core.util.types import default, as_tuple, native

_default_color_names = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']


def named_palette(name: Optional[str]=None, size: Optional[int]=None, desat=None):
    """
    returns a named standard palette using seaborn, if no name is specified the ts2d default palette is returned
    :param name: name of the standard palette
    :param
    """
    if name is None or name in {'ts2d', 'default'}:
        return default_palette(size=size)
    import seaborn as sns
    return list(np.multiply(np.clip(v, a_min=0, a_max=1.0), 255).astype(int)
                for v in sns.color_palette(name, size, desat))


def default_palette(size: Optional[int]):
    """
    returns an default arbitrary length color palette, after the last entry colors are generated randomly (with the same seed)
    """
    global _default_color_names
    size = default(size, len(_default_color_names))
    return list(default_color(i) for i in range(size))

def default_color(index: int) -> tuple:
    """
    returns a default color for the given index, after the last entry colors are generated randomly (with the same seed)
    """
    global _default_color_names, _default_color_cache
    assert index >= 0

    if index < len(_default_color_names):
        return to_color(_default_color_names[index])
    else:
        c = next(random_colors(index))
        return c

def random_colors(seed: int=0):
    """
    generator for random colors
    """
    import random
    rnd = random.Random(seed)
    while True:
        yield tuple(rnd.randint(32, 200) for _ in range(3))

def _name_to_color(name: str):
    from matplotlib import colors
    return colors.to_rgb(name)

def to_color(v: Union[int, tuple, list, str]):
    """
    returns a uint rgb value for a label int, rgb tuple or color name
    """
    if isinstance(v, str):
        v = _name_to_color(v)
    elif np.isscalar(v):
        if isinstance(v, int):
            return default_color(v)
        v = (v, ) * 3
    return tuple_to_color(v)

def tuple_to_color(v: tuple):
    v = as_tuple(v)
    n = len(v)
    if n != 3:
        raise RuntimeError("Color tuples need to be of length 3 (found: {})".format(n))
    if any(not isinstance(c, int) for c in v):
        v = np.multiply(np.clip(v, a_min=0, a_max=1.0), 255).astype(int)
    else:
        v = np.clip(v, a_min=0, a_max=255).astype(int)
    return native(v)

def to_color_str_rgb_floats(v: Union[int, tuple, list, str], sep=', '):
    v = to_color(v)
    v = native(np.clip(np.asarray(v, dtype=float) / 255, a_min=0, a_max=1))
    v = format_array(v, p=3, sep=sep)
    return v

def to_palette(v: Union[dict, list]):
    if isinstance(v, dict):
        if any(not isinstance(k, int) or k < 0 for k in v.keys()):
            raise RuntimeError("Dictionary palette must consist of non-negative integer keys!")
        lim = max(v.keys()) if v.keys() else 0
        res = list()
        res.append([255, 255, 255]) # background should be replaced
        for idx in range(1, lim+1):
            c = v.get(idx)
            if c is not None:
                c = to_color(c)
            else:
                c = default_color(idx)
            res.append(c)
        return res
    else:
        return list(to_color(c) for c in v)

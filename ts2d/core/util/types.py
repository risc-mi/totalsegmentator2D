from typing import Optional, Any, Mapping

import numpy as np


def default(val: Optional[Any], d: Any):
    """
    returns a default value if val is not set (None) or otherwise val
    :param val: value to check
    :param d: default value
    """
    return d if val is None else val


def as_set(a) -> set:
    """
    Convert an item which may be a container or a scalar to a set
    """
    if hasattr(a, '__iter__') and not isinstance(a, str):
        return set(tuple(e) if isinstance(e, list) else e
                   for e in a)
    return {a} if a is not None else set()


def as_tuple(a) -> tuple:
    """
    Convert an item which may be a container or a scalar to a tuple
    """
    if hasattr(a, '__iter__') and not isinstance(a, str):
        return tuple(a)
    return (a, ) if a is not None else tuple()


def as_list(a) -> list:
    """
    Convert an item which may be a container or a scalar to a list
    """
    if hasattr(a, '__iter__') and not isinstance(a, str):
        return list(a)
    return [a] if a is not None else []


def native(val, dtype=None, force=False):
    """
    Converts value from a numpy to native python type if applicable, otherwise pass through.
    :param val: value to convert
    :param dtype: optional dtype to cast to
    :param force: convert also if val is not a numpy type
    :return: 'native' python value
    """
    if hasattr(val, 'dtype'):
        if np.isscalar(val):
            return val.item() if dtype is None else val.astype(dtype).item()
        return np.asarray(val, dtype=dtype).tolist()
    elif force:
        return native(np.asarray(val, dtype=dtype))
    return val


def dict_get(d: dict, key: str, default=None, required=False, dtype=None):
    """
    reads the value at key from the specified dictionary, which can be nested
    key can specify a parameter or parameter group at an arbitrary level separated by '.'
    e.g., 'level1.level2.a'.
    Note: Nodes may have a leaf value and a nested subgroup, i.e., { 'a': 1, 'a.b': 2 }. In this case, the leaf value
    will be assigned to 'a.~' and accessing 'a' will return the leaf value if present. To explicitly access the leaf
    the notation 'a.~' can always be used, regardless of whether a subgroup exists.
    To explicitly access the subgroup, 'a.' can be used.

    :param d: the dictionary to read from
    :param key: key to access
    :param default: default value if the key does not exist
    :param required: whether to fail if key is missing (ignores the default value)
    :param dtype: type to convert the value to (using generic_convert)
    :return: value at key in the dictionary
    """
    convert = dtype
    if key not in d:
        lvls = key.split('.')
        stop = len(lvls)
        info = nest_dict(d, stop=stop)
        lvl = ''
        for lvl in lvls:
            if isinstance(info, Mapping):
                info = info if lvl == '' else info.get(lvl)
            else:
                if lvl != '~':
                    info = None

        if info is not None:
            # explicitely reference all subvalues
            if lvl != '':
                # check whether the dictionary has a root value, which is stored with the value '~'
                # otherwise the full sub dictionary is returned
                if isinstance(info, dict):
                    info = info.get('~', info)
            v = info
        else:
            if required:
                raise RuntimeError("Required parameter is missing: {}".format(key))
            v = default
            convert = None  # do not convert the default value
    else:
        v = d[key]
    if convert is not None:
        v = generic_convert(v, convert)
    return v


def _convert_nested_sequences(data: dict, fail: bool, path=''):
    if isinstance(data, _nested_sequence):
        if fail:
            if any(i not in set(data.keys()) for i in range(len(data))):
                raise RuntimeError("Sequence is missing indices: {}".format(path))
        res = list()
        for subk in sorted(data.keys()):
            spath = path + '[{}]'.format(subk)
            v = data[subk]
            v = _convert_nested_sequences(v, fail=fail, path=spath)
            res.append(v)
        data = res
    elif isinstance(data, dict):
            for k in data.keys():
                k = str(k)
                spath = path + '.' + k if path != '' else k
                data[k] = _convert_nested_sequences(data[k], fail=fail, path=spath)
    return data


def generic_convert(v: Any, dtype):
    from typing import get_args
    if hasattr(dtype, '__origin__'):
        otype = dtype.__origin__
        if otype == dict:
            tk, tv = get_args(dtype)
            eitems = dict(v).items()
            v = dict()
            for ek, ev in eitems:
                ek = generic_convert(ek, tk)
                ev = generic_convert(ev, tv)
                v[ek] = ev
        elif otype in [list, tuple]:
            tv = get_args(dtype)[0]
            v = otype(generic_convert(ev, tv)
                     for ev in v)
    else:
        if np.issubdtype(dtype, np.number):
            if isinstance(v, str):
                v = v.strip()
        v = dtype(v)
    return v


def is_container(v):
    """
    returns True if the value is a
    """
    return not isinstance(v, str) and hasattr(v, '__iter__')

def unwrap_singular(v, fail=True):
    """
    returns the singular element of a container
    if fail is True, the method raises an exception for any container with more than one element
    otherwise the original container is returned
    non-container types are passed through
    :param v: container element
    :param fail: whether to fail for containers with multiple elements (default behaviour), otherwise return the container
    :return: singular element of the container
    """
    if is_container(v):
        if len(v) == 1:
            if hasattr(v, 'values'):
                return next(iter(v.values()))
            return next(iter(v))
        elif fail:
            raise ValueError("Container does not contain exactly one element.")
    return v


def nest_dict(data: dict, check_sequence=False, stop=None):
    """
    returns a tree of nested dicts from a flat tree dictionary with '.' separated levels
    :param data: dictionary to nest
    :param check_sequence: whether to fail if sequences miss indices
    :param stop: level to stop nesting at
    :return: nested tree dictionary
    """
    res = dict()
    for k, v in data.items():
        parts = list(p.strip().lower() for p in k.split('.'))
        if any(len(p)==0 for p in parts):
            raise RuntimeError("Invalid key in tree dictionary: {}".format(k))

        d = res
        for pidx, part in enumerate(parts):
            if pidx == stop:
                d['.'.join(parts[pidx:])] = v
                break
            else:
                last = pidx == len(parts)-1
                name = part
                rval = None
                try:
                    rbx = part.index('[')
                    try:
                        rex = part.index(']')
                    except:
                        raise RuntimeError("Unable to nest dictionary: Nested list has closing bracket ']' after '[' (key: {})".format(k))
                    name = part[:rbx]
                    if len(name.strip()) == 0:
                        raise RuntimeError("Unable to nest dictionary: Sequence name cannot be empty!")
                    rval = part[rbx+1:rex]
                    try:
                        rval = int(rval)
                    except:
                        raise RuntimeError("Unable to nest dictionary: Nested list index is no integer: {} (key: {})".format(rval, k))
                except ValueError:
                    pass

                if rval is not None:
                    d = d.setdefault(name, _nested_sequence())
                    if not isinstance(d, _nested_sequence):
                        raise RuntimeError("Unable to nest dictionary: Key {} used as a sequence, but also found {}".format(k, type(d).__name__))
                    if last:
                        d[rval] = v
                    else:
                        sub = d.setdefault(rval, dict())
                        d = sub
                else:
                    if last:
                        sub = d.get(name)
                        if sub is not None:
                            if name != '~' and not isinstance(sub, dict):
                                # overrides the old leaf node
                                d[name] = dict()
                                d = d[name]
                            d['~'] = v
                        else:
                            d[name] = v
                    else:
                        sub = d.setdefault(name, dict())
                        if not isinstance(sub, dict):
                            # convert into leaf node
                            d[name] = {'~': sub}
                            sub = d[name]
                        d = sub

    # convert sequences
    res = _convert_nested_sequences(res, fail=check_sequence)

    return res


class _nested_sequence(dict):
    pass


def dict_merge(a: dict, b: dict, combine=False):
    """
    merges two dictionaries a and b while considering sub-dictionaries
    any other types are overwritten with the respective value in b
    :combine: if True, different values are not overridden but combined in a set, defaults to False
    :return: merged dictionary
    """
    keys = set(a.keys()).union(b.keys())
    res = dict()
    for k in keys:
        if k in a:
            if k in b:
                # merge subdictionaries only and override anything else with b[k]
                if isinstance(a[k], dict):
                    res[k] = dict_merge(a[k], b[k])
                elif combine:
                    # convert to sets, also make sure the underlying values can be hashed, therefore convert them to tuples
                    sa = as_set(as_tuple(a[k]))
                    sb = as_set(as_tuple(b[k]))
                    res[k] = unwrap_singular(sa.union(sb), fail=False)
                else:
                    res[k] = b[k]
            else:
                res[k] = a[k]
        else:
            res[k] = b[k]
    return res
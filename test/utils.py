import os
from typing import List

from ts2d.core.util.path import get_package_root
from ts2d.core.util.types import as_list


def get_asset_path(filename: str) -> str:
    res = os.path.abspath(os.path.join(get_package_root(), '..', 'assets', filename))
    assert os.path.exists(res), f"Asset file {filename} does not exist at {res}"
    return res


def assert_exist(fns: List[str] | str, dir=None):
    fps = as_list(fns)
    if dir is not None:
        fps = [os.path.join(dir, fn) for fn in fps]
    for fp in fps:
        assert os.path.exists(fp), f"Expected file {fp} does not exist!"
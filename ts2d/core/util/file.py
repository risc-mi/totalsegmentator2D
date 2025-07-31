import json
import os
import traceback
from typing import Union


def read_json(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as ex:
        raise RuntimeError(f"Failed to load json data from file: {path} ({ex})")



def write_json(data: Union[dict, list], path: str):
    with open(path, 'w') as f:
        return json.dump(data, f)

def enumerate_files(root: str):
    """
    recursively enumerates all files in root and its subfolders
    """
    for root, dirs, files in os.walk(root):
        if not os.path.basename(root).startswith('_'):
            for file in files:
                yield os.path.join(root, file)
        else:
            dirs.clear()
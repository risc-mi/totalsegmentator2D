import os
import pandas as pd

from ts2d.core.util.file import read_json
from ts2d.core.util.path import get_package_data_root


_CONFIG = None
_SHARED_URLS = None
_LABEL_COLORS = None

def get_label_colors():
    global _LABEL_COLORS
    if _LABEL_COLORS is None:
        fp = os.path.join(get_package_data_root(), 'label-colors.csv')
        _LABEL_COLORS = pd.read_csv(fp, index_col='Label')['Color'].to_dict()
    return _LABEL_COLORS

def get_shared_urls():
    global _SHARED_URLS
    if _SHARED_URLS is None:
        fp = os.path.join(get_package_data_root(), 'shared.json')
        _SHARED_URLS = read_json(fp)
    return _SHARED_URLS

def get_config():
    global _CONFIG
    if _CONFIG is None:
        fp = os.path.join(get_package_data_root(), 'config.json')
        _CONFIG = read_json(fp)
    return _CONFIG

def get_default_model():
    return get_config()['default-model']

if __name__ == '__main__':
    print(get_label_colors())
    print(get_shared_urls())
    print(get_config())
    print(get_default_model())
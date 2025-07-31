import os
import pandas as pd

from ts2d.core.util.file import read_json
from ts2d.core.util.log import warn
from ts2d.core.util.path import get_package_data_root


_CONFIG = None
_SHARED_URLS = None
_LABEL_COLORS = None

def get_label_colors():
    global _LABEL_COLORS
    if _LABEL_COLORS is None:
        fp = os.path.join(get_package_data_root(), 'label-colors.csv')
        colors = pd.read_csv(fp, index_col='Label')['Color'].to_dict()
        colors = dict((k.lower(), v) for k, v in colors.items())
        _LABEL_COLORS = colors
    return _LABEL_COLORS


def _get_github_shared_urls():
    try:
        url = "https://raw.githubusercontent.com/risc-mi/totalsegmentator2D/refs/heads/main/ts2d/data/shared.json"
        import requests
        r = requests.get(url)
        if not r.ok:
            raise RuntimeError(r.reason)
        import json
        return json.loads(r.text)
    except Exception as e:
        warn("Failed to load the shared URLs from GitHub ({}).\nUsing the URLs from the current checkout.".format(e))
        return None

def get_shared_urls(fetch_from_repo: bool=True):
    global _SHARED_URLS
    if _SHARED_URLS is None:
        # first try to get the latest information from the repository
        if fetch_from_repo:
            _SHARED_URLS = _get_github_shared_urls()
        if _SHARED_URLS is None:
            # ok, lets work with what we checked out
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

def get_test_model_single():
    return get_config()['default-test-model-single']

def get_test_model_multi():
    return get_config()['default-test-model-multi']

def get_model_resolve_map():
    return get_config()['default-resolve']

if __name__ == '__main__':
    print(get_label_colors())
    print(get_shared_urls())
    print(get_config())
    print(get_default_model())
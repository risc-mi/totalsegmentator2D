import glob
import os
import pytest

from ts2d.core.inference.database import URLDataBase
from ts2d.core.inference.zoo import NNUZoo
from ts2d.core.util.config import get_test_model_single, get_shared_urls
from ts2d.core.util.temp import SafeTemporaryDirectory


@pytest.fixture
def zoo():
    yield NNUZoo(remote=URLDataBase(get_shared_urls(fetch_from_repo=False)))

def test_download(zoo):
    remote = zoo.remote
    key = get_test_model_single()
    assert remote.latest(key=key) is not None, f"Failed to query latest version of the default model!"
    assert remote.has(key=key), f"The remote database does not contain the default model {key}!"

    with SafeTemporaryDirectory() as tmp:
        remote.copy(dest_root=tmp, key=key)
        assert glob.glob(os.path.join(tmp, "**", "model.json"), recursive=True), f"The downloaded model {key} does not contain a model.json file!"
        assert glob.glob(os.path.join(tmp, "**", "checkpoint_*.pth"), recursive=True), f"The downloaded model {key} does not contain a checkpoint!"


def test_zoo(zoo):
    key = get_test_model_single()
    zoo.resolve(key)
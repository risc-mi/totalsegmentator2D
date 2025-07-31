import glob
import os
import pytest

from ts2d.core.inference.zoo import NNUZoo
from ts2d.core.util.config import get_test_model
from ts2d.core.util.temp import SafeTemporaryDirectory


@pytest.fixture
def zoo():
    yield NNUZoo()

def test_download(zoo):
    remote = zoo.remote
    key = get_test_model()
    assert remote.latest(key=key) is not None, f"Failed to query latest version of the default model!"
    assert remote.has(key=key), f"The remote database does not contain the default model {key}!"

    with SafeTemporaryDirectory() as tmp:
        remote.copy(dest_root=tmp, key=key)
        assert glob.glob(os.path.join(tmp, "**", "model.json"), recursive=True), f"The downloaded model {key} does not contain a model.json file!"
        assert glob.glob(os.path.join(tmp, "**", "checkpoint_*.pth"), recursive=True), f"The downloaded model {key} does not contain a checkpoint!"


def test_zoo(zoo):
    key = get_test_model()
    zoo.access(key)
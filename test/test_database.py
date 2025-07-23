import glob
import os

from ts2d.core.inference.zoo import NNUZoo
from ts2d.core.util.config import get_default_model
from ts2d.core.util.temp import SafeTemporaryDirectory


def main():
    remote = NNUZoo().remote

    key = get_default_model()
    assert remote.latest(model=key) is not None, f"Failed to query latest version of the default model!"
    assert remote.has(key), f"The remote database does not contain the default model {key}!"

    with SafeTemporaryDirectory() as tmp:
        remote.copy(dest_root=tmp, model=key)
        assert glob.glob(os.path.join(tmp, "**", "model.json"), recursive=True), f"The downloaded model {key} does not contain a model.json file!"
        assert glob.glob(os.path.join(tmp, "**", "checkpoint_*.pth"), recursive=True), f"The downloaded model {key} does not contain a checkpoint!"
    print("Test successful!")

if __name__ == '__main__':
    main()
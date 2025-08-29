import os
import shutil
from tempfile import TemporaryDirectory

from test.utils import get_asset_path, assert_exist
from ts2d.core.util.config import get_test_model_single, get_test_model_multi


def test_single_basic():
    sample = "sample_s0521"
    with TemporaryDirectory() as tmp:
        fp = get_asset_path(f"{sample}.nrrd")
        ret = os.system(f"ts2d -i {fp} -o {tmp} --model {get_test_model_single()} --no-fetch")
        assert ret == 0, f"The CLI command failed with a non-zero exit code: {ret}"
        assert_exist([f"{sample}.seg.nrrd"], dir=tmp)


def test_single_default():
    # all other tests use a specific test model id and the --no-fetch flag to avoid fetching the latest model URLs from main
    # (which may not be up to date) instead, we use any default model for the cardiac group, which should always be available
    sample = "sample_s0521"
    with TemporaryDirectory() as tmp:
        fp = get_asset_path(f"{sample}.nrrd")
        ret = os.system(f"ts2d -i {fp} -o {tmp} --model ts2d_cardiac")
        assert ret == 0, f"The CLI command failed with a non-zero exit code: {ret}"
        assert_exist([f"{sample}.seg.nrrd"], dir=tmp)

def test_single_full():
    sample = "sample_s0521"
    with TemporaryDirectory() as tmp:
        fp = get_asset_path(f"{sample}.nrrd")
        ret = os.system(f"ts2d -i {fp} -o {tmp} --visualize --save-all --model {get_test_model_multi()} --no-fetch")
        assert ret == 0, f"The CLI command failed with a non-zero exit code: {ret}"

        fns = [f"{sample}.seg.nrrd", f"{sample}.seg.png",
               f"{sample}_max.nrrd", f"{sample}_mean.nrrd"]
        for group in ['cardiac', 'muscles', 'organs', 'ribs', 'vertebrae']:
            fns.extend([f"{sample}-{group}.seg.nrrd", f"{sample}-{group}.seg.png"])
        assert_exist(fns, dir=tmp)

def test_folder_basic():
    samples = ["sample_s0332", "sample_s0521"]
    with TemporaryDirectory() as tmp:
        dir_in = os.path.join(tmp, "input")
        dir_out = os.path.join(tmp, "output")

        os.makedirs(dir_in, exist_ok=True)
        for sample in samples:
            shutil.copy(get_asset_path(f"{sample}.nrrd"), dir_in)

        ret = os.system(f"ts2d -i {dir_in} -o {dir_out} --model {get_test_model_single()} --no-fetch")
        assert ret == 0, f"The CLI command failed with a non-zero exit code: {ret}"

        fns = list(f"{sample}.seg.nrrd" for sample in samples)
        assert_exist(fns, dir=dir_out)
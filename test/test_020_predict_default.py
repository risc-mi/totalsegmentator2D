from time import time, sleep

from test.utils import get_asset_path
from ts2d import TS2D
from ts2d.core.util.config import get_test_model_single
from ts2d.core.util.image import read_image
import SimpleITK as sitk
import pytest

@pytest.fixture
def model():
    key = get_test_model_single()
    model = TS2D(key=key, fetch_remote=False)
    yield model
    model.close()

def _general_test(model: TS2D, sample_name: str):
    fp = get_asset_path(sample_name)
    img = read_image(fp)

    print("Model initialized, starting prediction...")
    t1 = time()
    res = model.predict(img)
    elapsed = time() - t1
    print(f"Prediction took {elapsed:.3f} seconds")

    assert res is not None, "The prediction failed!"
    assert isinstance(res.get_segmentation(), sitk.Image), "Prediction contains no segmentation image!"

def test_2d_sample(model):
    """
    basic test for a 2D sample, which has already been projected
    """
    _general_test(model, 'sample_s0332.nrrd')

def test_3d_sample(model):
    """
    basic test for a 3D sample
    """
    _general_test(model, 'sample_s0521.nrrd')
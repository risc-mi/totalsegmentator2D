import os
from tempfile import TemporaryDirectory
from time import time, sleep

from test.utils import get_asset_path
from ts2d import TS2D
from ts2d.core.util.config import get_test_model_single
from ts2d.core.util.image import read_image
import SimpleITK as sitk
import pytest

@pytest.fixture
def result():
    with TS2D(key=get_test_model_single(), fetch_remote=False) as model:
        img = read_image(get_asset_path('sample_s0521.nrrd'))
        return model.predict(img)

def test_general(result):
    assert result is not None, "The prediction returned no result!"
    assert len(result.models) > 0, "The prediction result has no model information!"
    assert result.get_segmentation(), "Prediction contains no final segmentation image!"

def test_export(result):
    with TemporaryDirectory() as tmp:
        name = 'test'
        result.save(dest=tmp, name=name, targets='all', models='all', ext='nrrd', content='all')
        fns = [f'{name}', f'{name}.seg'] + list(f'{name}_{ch}' for ch in result.get_projection().keys())
        fns = [f'{fn}.{ext}' for ext in ('nrrd', 'png') for fn in fns]
        for fn in fns:
            assert os.path.exists(os.path.join(tmp, fn)), f"Expected export file '{fn}' was not found after export!"

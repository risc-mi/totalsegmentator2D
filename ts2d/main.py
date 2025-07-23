import SimpleITK as sitk

from ts2d.core.util.config import get_label_colors
from ts2d.core.util.image import project, read_image, reduce_dimensions
from ts2d.core.util.types import dict_get, unwrap_singular


class TS2D:
    def __init__(self):
        colors = get_label_colors()
        param = {
            'server.workers': 1,
            'nnu.result.colors': colors
        }

        from ts2d.core.inference.zoo import NNUZoo
        self.model = NNUZoo().load("ts2d_v1_ep4000b2", interface="process", param=param)
        self.model.start()

    def _load_models(self, name: str):
        pass


    @staticmethod
    def _project(img: sitk.Image, mode: str):
        res = project(img, mode=mode, axis='coronal')
        return reduce_dimensions(res)

    def predict(self,
                input: sitk.Image | str,
                ofile: str | None = None):

        if isinstance(input, str):
            input = read_image(input)
        if not isinstance(input, sitk.Image):
            raise RuntimeError(f"input must be a string path or a SimpleITK image, found: {type(input).__name__}")

        if input.GetDimension() > 2:
            channels = dict_get(self.model.config, 'inputs')
            input = list(self._project(input, mode=m) for m in channels)
            if len(input) > 1:
                input = sitk.Compose(input)
            else:
                input = unwrap_singular(input)

        res = self.model.apply(input)
        if ofile:
            sitk.WriteImage(res, ofile)
        return res
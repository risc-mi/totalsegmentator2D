import os
import traceback
from typing import List

import SimpleITK as sitk

from ts2d.core.inference.database import decompose_model_key, URLDataBase
from ts2d.core.inference.nnu import NNUProcessModel, NNUModel
from ts2d.core.util.config import get_label_colors, get_shared_urls
from ts2d.core.util.image import project, reduce_dimensions, read_image, create_visual, reorient_image, \
    get_actual_dimension, combine_segmentations, split_channels
from ts2d.core.util.log import warn, log
from ts2d.core.util.meta import copy_image_meta, copy_image_geo
from ts2d.core.util.types import unwrap_singular, as_set, as_list
from ts2d.core.util.util import mkdirs
from ts2d.core.inference.zoo import NNUZoo


class TS2D:
    def __init__(self, key: str="ts2d",
                 use_remote: bool=True,
                 fetch_remote: bool=True):
        """
        Initialize the TS2D instance with the specified model key.
        :param key: the model key to use, defaults to "ts2d"
        :param use_remote: whether to allow the use of remote models, defaults to True. If False, only local models will be used.
        :param fetch_remote: whether to fetch the latest URLs from the repository (shared.json) instead of the locally checked out version
        """
        colors = get_label_colors()
        param = {
            'server.workers': 1,
            'nnu.result.colors': colors
        }

        # initialize the models matching to the key
        remote = URLDataBase(get_shared_urls(fetch_remote)) if use_remote else False
        self.zoo = NNUZoo(remote=remote)
        self.models = dict()
        ids = self.zoo.resolve(key, unique_model=True)
        if len(ids) > 1:
            log(f"The model key '{key}' was resolved to {len(ids)} models: {', '.join(ids)}.")
        for id in ids:
            try:
                model = self.zoo.load(id, interface="process", param=param)
                model.start(wait=False)
                if not model.multilabel:
                    warn(f"The loaded model {id} is not configured for multilabel inference - this should not be the case in TS2D and may lead to unexpected results.")
                self.models[id] = model
            except:
                traceback.print_exc()
                raise RuntimeError(f"Failed to load model {id}" + (f" (resolved from {key})" if key != id else ""))
        for model in self.models.values():
            if isinstance(model, NNUProcessModel):
                model.await_startup()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def close(self):
        for model in self.models.values():
            if isinstance(model, NNUProcessModel):
                model.stop()
        self.models = dict()


    def __del__(self):
        if self.models:
            warn("The TS2D instance is being deleted without calling close() - cleaning up all models. "
                 "Call close() explicitly before deleting the instance to avoid concurrency issues.")
            try:
                self.close()
            except:
                traceback.print_exc()
                warn("Failed to clean up models on deletion - the exception was ignored."
                     "Call close() explicitly before deleting the instance to avoid issues.")

    def predict(self,
                input: sitk.Image | str,
                collapse: bool = False,
                merge: bool = True):
        """
        Predict the segmentation for the given input image.
        :param input: the input image, either a SimpleITK image or a path to an image file
        :param collapse: whether to collapse the dimensions of the output image and segmentation to 2D, this discards the 3D orientation
        :param merge: whether to merge the individual segmentations from the models into a single segmentation image.
        :return: a dictionary of results:
        - 'input': the input image (SimpleITK image)
        - 'segmentation': the predicted segmentation (SimpleITK image)
        - 'success': whether the prediction was successful
        - 'paths.segmentation': the path to the saved segmentation file (if odir is specified)
        - 'paths.visuals.segmentation': the path to the saved visualization of the segmentation (if visualize is True)
        - 'paths.visuals.input[n]': the path to the saved visualization of the input channel n (if visualize is True)
        """

        if isinstance(input, str):
            input = read_image(input)
        if not isinstance(input, sitk.Image):
            raise RuntimeError(f"input must be a string path or a SimpleITK image, found: {type(input).__name__}")

        result = dict()
        cache = dict()
        for id, model in self.models.items():
            res = self._predict_model(id, input=input, collapse=collapse, cache=cache)
            result.setdefault('models', dict())[id] = res

        if merge:
            res = dict((id, res['segmentation']) for id, res in result['models'].items())
            if len(res) == 1:
                # only one model, return the segmentation directly
                result['segmentation'] = unwrap_singular(res.values())
            else:
                # merge the segmentations into a single image
                segs: List[sitk.Image] = list(res.values())
                result['segmentation'] = combine_segmentations(segs)

        result['input'] = input
        projections = cache.get('projections')
        if projections:
            result['projections'] = projections

        return TS2D.Result(result)


    def _predict_model(self, id: str, input, collapse: bool, cache: dict):
        """
        Predict the segmentation for the specified model id
        """

        model = self.models.get(id)
        assert isinstance(model, NNUModel), f"Model with id '{id}' is not available."

        result = dict()
        result['id'] = id
        result['model'], result['group'] = decompose_model_key(id)
        result['revision'] = model.revision

        channels = model.channels
        if channels is None:
            raise RuntimeError(f"Model {id} does not have a channel definition, cannot project the input image.")
        channels = sorted(channels.items(), key=lambda x: x[0])

        # create the model input
        projections = cache.setdefault('projections', dict())
        if get_actual_dimension(input) > 2:
            # project the 3D input image to 2D and use the requested projection method for each channel
            input = reorient_image(input, orient='RAI')
            ch_list = list()
            for ch_idx, ch_name in channels:
                if ch_name not in projections:
                    projections[ch_name] = self._project(input, mode=ch_name)
                ch_list.append(projections[ch_name])
            input = sitk.Compose(ch_list) if len(ch_list) > 1 else unwrap_singular(ch_list)
        else:
            model_nch = len(channels)
            input_nch = input.GetNumberOfComponentsPerPixel()
            if model_nch != input_nch:
                raise RuntimeError(f"The number of channels in the input image does not match the models "
                                   f"channel definition ({model_nch} vs {input_nch}).")
            projections.update((f"ch{ch_idx}", ch) for ch_idx, ch in enumerate(split_channels(input)))

        # the actual inference
        native_2d = input.GetDimension() < 3
        input2D = input if native_2d else reduce_dimensions(input)
        seg = model.apply(input2D)
        assert isinstance(seg, sitk.Image), f"Model returned an unexpected result: expected a segmentation image and found {type(seg).__name__}."
        seg = seg if collapse or native_2d else self._restore_dimension(seg, input)
        input = input2D if collapse else input

        result['input'] = input
        result['segmentation'] = seg
        return result

    @staticmethod
    def _project(img: sitk.Image, mode: str, pixelType: str = sitk.sitkFloat32):
        res = project(img, mode=mode, axis='coronal')
        res = sitk.Cast(res, pixelType)
        return res

    @staticmethod
    def _restore_dimension(img: sitk.Image, ref: sitk.Image):
        import numpy as np
        img = copy_image_meta(
            sitk.GetImageFromArray(np.reshape(sitk.GetArrayFromImage(img), np.flip(ref.GetSize()).tolist() + [-1])),
            img)
        return copy_image_geo(img, ref)

    class Result:
        def __init__(self, data: dict):
            self.data = data

        @property
        def models(self) -> List[str]:
            """
            returns a list of model ids used in the prediction
            """
            return sorted(self.data.get('models', dict()).keys())

        def get_input(self, model: str | None=None):
            """
            returns either the initial input image or the processed input for a specific model
            Note: the initial input image is taken before the projection to 2D
            :param model: the model id to get the input for, if None, returns the initial input image
            """
            if model is not None:
                return self.data.get('models', dict()).get(model, dict()).get('input')
            return self.data.get('input')

        def get_segmentation(self, model: str | None=None):
            """
            returns either the final combined segmentation or the segmentation for a specific model
            :param model: the model id to get the segmentation for, if None, returns the final combined segmentation
            """
            if model is not None:
                return self.data.get('models', dict()).get(model, dict()).get('segmentation')
            return self.data.get('segmentation')

        def get_projection(self, channel: str | None=None):
            """
            returns the projection for a specific channel or a dictionary of all projections if channel is None
            :param channel: the channel name to get the projection for, if None, returns all projections
            """
            projections = self.data.get('projections', dict())
            if channel is not None:
                return projections.get(channel)
            return projections

        def save(self,
                 dest: str, name: str = 'result', ext: str = 'nrrd',
                 models: str | List[str] = 'all',
                 targets: str | List[str] = 'all',
                 content: str = 'all',
                 naming: str = 'group'):
            """
            Saves the result to the specified directory.
            :param dest: the output directory to save the results to, if None, no results will be saved
            :param name: the name of the output file, defaults to 'result'
            :param ext: the file extension for the output files, defaults to 'nrrd' (does not affect PNG visualizations)
            :param targets: one or many targets to export, can be a list or 'all', 'input', 'segmentation', 'projection'
            :param models: one or many model ids to export, can be 'all' to export all models, 'final' refers to the final combined target
            :param content: export type, can be 'file', 'visual' or 'all', defaults to 'file'
            :param naming: the naming scheme to use for the output files, can be 'group' or 'model'
            """
            assert ext.lower() != 'png', "PNG is not a valid export format for the 'file' content type."
            assert naming in {'group', 'model'}, f"Invalid naming scheme '{naming}', must be one of 'group' or 'model'."
            assert content in {'file', 'visual', 'all'}, f"Invalid export type '{content}', must be one of 'file', 'visual' or 'all'."
            content = {'visual', 'file'} if content == 'all' else {content}

            models = as_set(t.strip().lower() for t in as_list(models))
            if 'all' in models:
                models |= set(self.models) | {None}
            if 'final' in models:
                models |= {None}
            targets = as_set(t.strip().lower() for t in as_list(targets))

            def _make_filename(base, key):
                if key is not None and naming == 'group':
                    _, group = decompose_model_key(key)
                    return f"{base}-{group}"
                return base

            def _write(img, filepath):
                sitk.WriteImage(img, filepath, True)

            def _export_image(img, base_name, suffix="", labels=False):
                if 'file' in content:
                    _write(img, os.path.join(dest, f"{base_name}{suffix}.{ext}"))
                if 'visual' in content:
                    if labels:
                        vis = create_visual(img, labels=labels, axis='coronal')
                        _write(vis, os.path.join(dest, f"{base_name}{suffix}.png"))
                    else:
                        nch = img.GetNumberOfComponentsPerPixel()
                        for cidx, ch in enumerate(split_channels(img)):
                            vis = create_visual(ch, labels=labels, axis='coronal')
                            file_name = f"{base_name}{suffix}.png" if nch == 1 else f"{base_name}-ch{cidx}{suffix}.png"
                            _write(vis, os.path.join(dest, file_name))


            mkdirs(dest)

            # Export input images
            if {'all', 'input'} & targets:
                for key in models:
                    img = self.get_input(key)
                    if img:
                        _export_image(img, _make_filename(name, key))

            # Export segmentations
            if {'all', 'segmentation'} & targets:
                for key in models:
                    img = self.get_segmentation(key)
                    if img:
                        _export_image(img, _make_filename(name, key), suffix=".seg", labels=True)

            # Export projections
            if {'all', 'projection'} & targets:
                for channel, img in self.get_projection().items():
                    base = f"{name}_{channel}"
                    if 'file' in content:
                        _write(img, os.path.join(dest, f"{base}.{ext}"))
                    if 'visual' in content:
                        vis = create_visual(img)
                        _write(vis, os.path.join(dest, f"{base}.png"))
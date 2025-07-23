import copy
import os
import traceback
from functools import partial
from typing import Optional, Union, List, Dict
import SimpleITK as sitk

from ts2d.core.inference.database import _describe_model, _revision_str
from ts2d.core.inference.wrapper import NNUWrapper
from ts2d.core.util.util import mkdirs, parse_int, removeprefix
from ts2d.core.util.types import dict_get
from ts2d.core.util.log import warn
from ts2d.core.util.temp import SafeTemporaryDirectory


class NNUModel:
    """
    Interface providing access to a pretrained model for inference
    """
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._root = str(config['root'])
        if not self.name.isidentifier():
            raise RuntimeError("Invalid model name \"{}\": no valid python identifier".format(self.name))
        if not self.revision.isidentifier():
            raise RuntimeError("Invalid model revision \"{}\": no valid python identifier".format(self.revision))

        self._param = dict()
        self._nnu: NNUWrapper = None

        param = self._config.get('param', dict()).copy()
        folds = self._config.get('folds')
        if folds:
            param['nnu.folds'] = folds
        self.update_param(param)

        self.nnu.verify_setup()

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @property
    def nnu(self):
        """
        The nnu wrapper instance
        """
        return self._nnu

    @property
    def uid(self):
        return "ts2d_{}_{}".format(self.name, self.revision)

    @property
    def name(self):
        return self._config['model']

    @property
    def revision(self):
        r = self._config['revision']
        return _revision_str(r) if isinstance(r, int) else r

    @property
    def folds(self):
        return tuple(self._config['folds'])

    def __str__(self):
        return _describe_model(model=self.name, revision=self.revision, folds=self.folds)

    def apply(self, inputs: Union[str, sitk.Image, List[str], Dict[str, str], Dict[str, sitk.Image]], result_dir: Optional[str]=None, override=False) -> Union[dict, str, sitk.Image]:
        """
        Apply the model to predict results for each input
        :param inputs: can be (a single/list/dictionary) of (paths, SimpleITK images)
        :param result_dir: if specified, results will be stored in the specified directory, otherwise the images are loaded and returned
        :param override: whether to override existing results, only applies if results_dir is set
        :return: returns either a single result or a dictionary of (named) results depending of one or many inputs have been specified,
                 images are returned as paths if result_dir is set, otherwise as SimpleITK images
        """
        raise NotImplementedError()

    def update_param(self, param: dict):
        """
        update the specified parameters for the model and reload it
        :param param: a dictionary of parameters and values to update
        """

        # make sure a custom trainer (if available) receives a unique id
        uid_val = self.uid if param.get('nnu.trainer.file') is not None else None
        if param.get('nnu.trainer.uid') != uid_val:
            if param.get('nnu.trainer.uid') is not None:
                warn('Parameter "nnu.trainer.uid" is set automatically and should not be specified in the configuration, '
                     'changing from "{}" to "{}"'.format(param['nnu.trainer.uid'], self.uid))
            param['nnu.trainer.uid'] = self.uid

        self._param.update(param)

        self._nnu = NNUWrapper(self._param)
        self._nnu.configure(result_dir=self._config['root'], data_dir=None, make_task=False, load_splits=False, override=False)


class NNUProcessModel(NNUModel):
    def __init__(self, config: dict):
        from ts2d.core.inference.predictor import ParallelPredictor
        self._predictor = ParallelPredictor()
        super().__init__(config)

    def update_param(self, param):
        super().update_param(param)
        self._predictor.labels = self.nnu.get_labels()
        self._predictor.colors = dict_get(self._param, 'nnu.result.colors', default='ts2d')

    def start(self):
        workers = self._param.get('server.workers', 4)
        predictor = self._get_predictor()
        self._predictor.start(predictor, num_workers=workers)

    def stop(self):
        try:
            self._predictor.stop()
        except:
            pass

    def _get_predictor(self):
        model = self.nnu['result.data.dir']
        checkpoint = self._param.get('nnu.predict.checkpoint', 'best')
        augment = self._param.get('nnu.predict.augment', False)
        stepsize = self._param.get('nnu.predict.stepsize', 1)
        verbose = self._param.get('nnu.verbose', False)
        return partial(self._lazy_load_predictor,
                       model=model, folds=self.folds, checkpoint=f'checkpoint_{checkpoint}.pth',
                       augment=augment, stepsize=stepsize, verbose=verbose)

    @staticmethod
    def _lazy_load_predictor(model, folds, checkpoint, augment, stepsize, verbose):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(tile_step_size=stepsize,
                                    use_gaussian=True,
                                    use_mirroring=augment,
                                    perform_everything_on_device=True,
                                    verbose=verbose)
        predictor.initialize_from_trained_model_folder(model, folds, checkpoint)
        return predictor

    @staticmethod
    def _apply_predictor(predictor,
                         inputs: Union[str, sitk.Image, List[str], Dict[str, str], Dict[str, sitk.Image]],
                         result_dir: Optional[str]=None,
                         override: bool=True,
                         wait: bool=True) -> Union[dict, str, sitk.Image, Dict[str, sitk.Image]]:
        try:
            with SafeTemporaryDirectory() as tmp:
                res_paths = None
                res_single = False


                if isinstance(inputs, (str, sitk.Image)):
                    inputs = [inputs]
                    res_single = True
                if isinstance(inputs, (list, tuple)):
                    inputs = dict((f'image{idx+1}', img)
                                  for idx, img in enumerate(inputs))


                input_dir = os.path.join(tmp, 'input')
                output_dir = os.path.join(tmp, 'output') if result_dir is None else result_dir
                mkdirs([input_dir, output_dir])

                ids = dict()
                results = dict()
                for name, img in inputs.items():
                    ofile = os.path.join(output_dir, f'{name}.nrrd')
                    if isinstance(img, sitk.Image):
                        fp_img = os.path.join(input_dir, f'{name}.nrrd')
                        sitk.WriteImage(img, fp_img)
                        path_input = False
                    elif isinstance(img, str):
                        fp_img = img
                        path_input = True
                    else:
                        raise RuntimeError("Invalid input type: {}".format(type(img)))
                    if (res_paths is not None) and (res_paths != path_input):
                        raise RuntimeError("Mixed input types are not supported, choose either images or paths!")
                    res_paths = path_input

                    if wait:
                        ofile = predictor.predict(fp_img, ofile, overwrite=override)
                        results[name] = ofile if res_paths else sitk.ReadImage(ofile)
                    else:
                        task_id = predictor.predict(fp_img, ofile, overwrite=override, wait=False)
                        ids[task_id] = name
                if ids:
                    for task_id, task in predictor.wait(ids.keys()).items():
                        if not task.success:
                            warn(f"Prediction failed: {task.error}")
                            raise RuntimeError(f"Prediction failed for: {task.ofile}")
                        name = ids[task_id]
                        results[name] = task.ofile if res_paths else sitk.ReadImage(task.ofile)

                if res_single:
                    return next(iter(results.values()))
                return results
        except:
            traceback.print_exc()
            raise

    def apply(self, inputs: Union[str, sitk.Image, List[str], Dict[str, str], Dict[str, sitk.Image]],
              result_dir: Optional[str]=None,
              override=True) -> Union[dict, str, sitk.Image, Dict[str, sitk.Image]]:
        """
        Apply the model to predict results for each input
        :param inputs: can be (a single/list/dictionary) of (paths, SimpleITK images)
        :param result_dir: if specified, results will be stored in the specified directory, otherwise the images are loaded and returned
        :param override: whether to override existing results, only applies if results_dir is set
        :return: returns either a single result or a dictionary of (named) results depending of one or many inputs have been specified,
                 images are returned as paths if result_dir is set, otherwise as SimpleITK images
        """
        return NNUProcessModel._apply_predictor(self._predictor, inputs, result_dir, override, wait=False)


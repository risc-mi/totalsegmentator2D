import glob
import os
import re
import shutil
import SimpleITK as sitk
from datetime import datetime
from typing import List, OrderedDict

from ts2d.core.util.color import named_palette
from ts2d.core.util.file import read_json
from ts2d.core.util.log import warn, log
from ts2d.core.util.util import mkdirs, format_array, removeprefix, parse_int
from ts2d.core.util.types import default, dict_get

def nnu_find_datasets(root: str, version=None, dowarn=False):
    version_prefixes = {
        1: 'Task',
        2: 'Dataset'
    }
    prefixes = [version_prefixes.get(version)] if version is not None else list(version_prefixes.values())
    dns = os.listdir(root)
    tasks = OrderedDict()
    for dn in dns:
        for prefix in prefixes:
            if dn.startswith(prefix):
                try:
                    id = int(removeprefix(dn, prefix).split('_')[0])
                    tasks[id] = dn
                except Exception as ex:
                    if dowarn:
                        warn("Failed to parse id from '{}': {}".format(dn, ex))
    return tasks


class NNUWrapper:
    """
    Wrapper to configure and represent the nnU-Net model in the ts2d framework.
    [Setup]
    - nnu.version: to restrict the version of nnu-net (1 or 2)
    - nnu.task: task id to reference
    - nnu.folds: folds to train (must be defined in the dataset split)
    - nnu.plans: plans to reference (default: 'nnUNetPlans')
    - nnu.configuration: trainer class to use (default: '3d_fullres')
    - nnu.verbose: set the verbose flag for nnU-Net calls
    - nnu.trainer: trainer to reference (default: 'nnUNetTrainer')

    - nnu.predict.augment: whether to use test-time augmentation for prediction, CAREFUL, as this may increase prediction time by factor of 8!
    - nnu.predict.stepsize: stepsize (conversely, the overlap of patches) to use during sliding-window prediction, 0.5 is the nnu default (SLOW), 1.0 elimates overlaps
    - nnu.predict.checkpoint: (optional) model checkpoint to use for prediction, best or final (defaults to final)
    - nnu.result.colors: colors to assign to labels in the predicted segmentation
    """

    def __init__(self, param: dict):
        self.task_name = None
        self.version = dict_get(param, 'nnu.version', default=2, dtype=int)
        self.task_id = dict_get(param, 'nnu.task', default=None, dtype=int)
        self.folds = dict_get(param, 'nnu.folds', default=None, dtype=List[int])
        self.plans = dict_get(param, 'nnu.plans', default='nnUNetPlans', dtype=str)
        self.configuration = dict_get(param, 'nnu.configuration', default='3d_fullres', dtype=str)
        self.verbose = dict_get(param, 'nnu.verbose', default=False, dtype=bool)
        self.multilabel = False

        self.trainer = dict_get(param, 'nnu.trainer', default='nnUNetTrainer', dtype=str)
        self.checkpoint_name = dict_get(param, 'nnu.predict.checkpoint', default='final', dtype=str)
        self.augment = dict_get(param, 'nnu.predict.augment', default=True, dtype=bool)
        self.stepsize = dict_get(param, 'nnu.predict.stepsize', default=None, dtype=float)

        self._result_colors = dict_get(param, 'nnu.result.colors', default='ts2d')

        self._config = None
        self._adapter = self._create_adaptee(self, self.version)

    @staticmethod
    def _create_adaptee(wrapper, version):
        adapteeType = {
            2: _NNUAdapterV2
        }.get(version)
        if adapteeType is None:
            raise RuntimeError("No implementation for specified nnu version: {}".format(version))
        return adapteeType(wrapper)

    def __getitem__(self, key: str):
        return self._config.get(key)

    def verify_setup(self):
        """
        The method checks whether the python environment is configured to run nnUNet
        If any precondition is not met, the method will yield a corresponding exception
        """
        try:
            try:
                import torch
            except:
                raise RuntimeError("Pytorch is not available in the current python environment!")
            if not torch.cuda.is_available():
                warn("CUDA is not available in the installed Pytorch package!", once=True)
        except:
            warn(
                "WARNING: The Pytorch package (pytorch/torch) is not correctly installed in the current python environment.\n"
                "--- INSTRUCTIONS ---\n"
                "Try reinstalling the package and ensure the correct CUDA version is available.\n"
                "Follow the instructions at https://pytorch.org/get-started/locally/#windows-python and make \n"
                "sure to install a version that has CUDA enabled (neither CPU or ROCm).\n"
                "When installing through pip, make sure to install the correct CUDA version in your OS (from https://developer.nvidia.com/cuda-toolkit-archive).\n"
                "Make sure not to have multiple version of pytorch installed (using conda list AND pip list)."
                "Also check the core dependencies of pytorch: torchvision and torchaudio.\n"
                "Note: the naming of the Pytorch package is inconsistent in Conda (pytorch) and Pip (torch).\n"
                "This affects the list commands, however not the imports.\n"
                "--------------------\n\n")
            raise
        self._adapter.verify_setup()

    def configure(self, result_dir: str):
        detected_tasks = nnu_find_datasets(result_dir, version=self.version)
        target_task_id, target_task_name = self._check_detected_tasks(detected_tasks, expected=self.task_id)
        trainer_name = '__'.join([self.trainer, self.plans, self.configuration])

        result_task_dir = os.path.join(result_dir, target_task_name)
        result_data_dir = os.path.join(result_task_dir, trainer_name)
        result_log_name = 'Log{}.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        result_log_path = os.path.join(result_task_dir, result_log_name)

        result_config_name = 'dataset.json'
        result_config_path = os.path.join(result_data_dir, result_config_name)
        result_config_data = result_img_ext = result_channels = None
        result_multilabel = False
        if os.path.exists(result_config_path):
            try:
                result_config_data = read_json(result_config_path)
                result_img_ext = self._adapter.get_configured_ext(result_config_data)
                result_channels = self._adapter.get_channels(result_config_data)
                result_multilabel = result_config_data.get('multilabel', result_config_data.get('multiclass', False))
            except:
                warn("Failed to read the result dataset.json at: {}".format(result_config_path))

        result_fold_dirs = list()
        if os.path.exists(result_data_dir):
            result_fold_dirs.extend(os.path.join(result_data_dir, p)
                                    for p in os.listdir(result_data_dir)
                                    if re.match("fold_[0-9]+", p))
        result_fold_ids = list(int(os.path.basename(p).split('_')[1])
                               for p in result_fold_dirs)

        self._config = {
            'result.dir': result_dir,
            'result.task.dir': result_task_dir,
            'result.data.dir': result_data_dir,
            'result.log.name': result_log_name,
            'result.log.path': result_log_path,
            'result.fold.ids': result_fold_ids,
            'result.fold.dirs': result_fold_dirs,
            'result.config.data': result_config_data,
            'result.config.name': result_config_name,
            'result.config.path': result_config_path,
            'result.img.ext': result_img_ext,
            'result.img.channels': result_channels,
            'target.task': target_task_id,
        }
        self.folds = default(self.folds, result_fold_ids)
        self.task_id = target_task_id
        self.task_name = target_task_name
        self.multilabel = result_multilabel

    def _read_dataset_config(self):
        """
        returns the contents of the dataset config file, either from the data or the result folder
        """
        path = self._config.get('data.config.path')
        path = self._config.get('result.config.path') if path is None else path
        if path is None:
            raise RuntimeError("The dataset configuration has not been discovered - run and check self.configure")
        try:
            return read_json(path)
        except Exception as ex:
            raise RuntimeError("Failed to load the dataset config: {}".format(ex))

    def _check_detected_tasks(self, tasks: dict, expected=None):
        if not tasks:
            raise RuntimeError("The specified task was not found - no existing tasks identified!")
        if expected is None:
            if len(tasks) > 1:
                raise RuntimeError(
                    "The task id is ambiguous as multiple tasks are available ({}): "
                    "specify 'nnu.task'".format(format_array(tasks)))
            expected = list(tasks.keys())[0]
        if expected not in tasks:
            raise RuntimeError("The selected task is not available: {}".format(expected))
        return expected, tasks[expected]

    def get_exts(self):
        """
        returns a list of supported image file extensions
        """
        return self._adapter.get_exts()

    def get_channels(self):
        """
        returns a list of supported image file extensions
        """
        return self['result.img.channels']

    def get_labels(self) -> dict:
        """
        returns a dictionary of labels configured with the model
        """
        dataset = self._read_dataset_config()
        return self._adapter.get_configured_labels(dataset)

    def get_colors(self) -> dict:
        colors = self._result_colors
        if isinstance(colors, str):
            labels = list(ln for lv, ln in sorted(self.get_labels().items(), key=lambda x: x[0]))
            colors = dict(zip(labels, named_palette(colors, len(labels))))
        return colors

    def get_dims(self) -> int:
        """
        returns the dimensionality of the model, 2d or 3d
        """
        return self._adapter.get_dims()


class _NNUAdapter:
    def __init__(self, wrapper: NNUWrapper):
        self.nnu = wrapper

    def verify_setup(self):
        raise NotImplementedError()

    def get_env_mapping(self):
        raise NotImplementedError()

    def get_exts(self):
        raise NotImplementedError()

    def get_dims(self) -> int:
        raise NotImplementedError()

    def get_configured_ext(self, config: dict):
        raise NotImplementedError()

    def get_channels(self, config: dict) -> dict:
        raise NotImplementedError()

    def get_configured_labels(self, config: dict) -> dict:
        raise NotImplementedError()


class _NNUAdapterV2(_NNUAdapter):
    def get_env_mapping(self):
        return {
            "nnUNet_raw": "data.dir",
            "nnUNet_preprocessed": "preprocessed.dir",
            "nnUNet_results": "result.dir"
        }

    def verify_setup(self):
        try:
            try:
                import nnunetv2
            except:
                raise RuntimeError("nnUNet is not available in the active python environment!")
        except:
            warn("ERROR: The nnUNet package (nnunetv2) is not correctly installed in the current python environment!")
            raise

    def get_configured_ext(self, config: dict):
        return str(config['file_ending']).lstrip('.')

    def get_configured_labels(self, config: dict) -> dict:
        return dict(enumerate(config['labels'].keys()))

    def get_channels(self, config: dict) -> dict:
        return dict((parse_int(c), n) for c, n in config['channel_names'].items())

    def get_exts(self):
        return ['png', 'bmp', 'nii.gz', 'nrrd', 'mha', 'tif', 'tiff']

    def get_dims(self) -> int:
        return 2 if ('2d' in str(self.nnu.configuration).lower()) else 3

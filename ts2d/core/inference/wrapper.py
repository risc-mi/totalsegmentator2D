import glob
import os
import re
import shutil
import SimpleITK as sitk
from datetime import datetime
from typing import List, OrderedDict

from ts2d.core.util.file import read_json
from ts2d.core.util.log import warn, log
from ts2d.core.util.util import mkdirs, format_array, removeprefix
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
    Documentation for param:
    file, even if the trainer class is already available. defaults to False. (checking for the nnu.trainer.uid)

    [General setup]
    - nnu.version: to restrict the version of nnu-net (1 or 2)
    - nnu.task: task id to reference
    - nnu.folds: folds to train (must be defined in the dataset split)
    - nnu.plans: plans to reference (default: 'nnUNetPlans')
    - nnu.configuration: trainer class to use (default: '3d_fullres')
    - nnu.verbose: set the verbose flag for nnU-Net calls

    [Trainer configuration]
    - nnu.trainer: trainer to reference (default: 'nnUNetTrainer'), for custom trainers, the name of the trainer used for training
    - nnu.trainer.file: (optional) a custom trainer file to install
    - nnu.trainer.name: (optional) class name of the custom trainer, defaults to the value of 'nnu.trainer'. Used to discover the class during trainer installation.
    - nnu.trainer.uid: (optional) for custom trainers only, use only for prediction when multiple revisions of the same trainer may exist! Unique id used to reference the trainer class, defaults to the value of 'nnu.trainer'.
    - nnu.trainer.update: (optional) install the trainer

    [Processing]
    - nnu.predict.cpu: (optional) use the CPU to predict, defaults to False
    - nnu.processes.general: parallel processes to use by default
    - nnu.processes.preprocessing: parallel process to use for preprocessing (i.e., reading images and extracting patches)
    - nnu.processes.export: parallel process to use for export (i.e., merging and writing predictions)
    - nnu.preprocessed.name: optional name of the folder to store the preprocessed dataset in the result group,
                             defaults to '_preprocessed', ignored if 'nnu.preprocessed.path' is set
    - nnu.preprocessed.path: optional folder to store the preprocessed dataset, should be located on an SSD,
                             setting this will override 'nnu.preprocessed.name'

    [Prediction]
    - nnu.predict.augment: whether to use test-time augmentation for prediction, CAREFUL, as this may increase prediction time by factor of 8!
    - nnu.predict.stepsize: stepsize (conversely, the overlap of patches) to use during sliding-window prediction, 0.5 is the nnu default (SLOW), 1.0 elimates overlaps
    - nnu.predict.probabilities: whether to write the predicted softmax probabilities (as npz files), CAREFUL, as this produces very large files
    - nnu.predict.orient: (optional) target orientation for input volumes (e.g., RAS, LPS, ...)
    - nnu.predict.checkpoint: (optional) model checkpoint to use for prediction, best or final (defaults to final)
    - nnu.predict.multichannel.split: whether to split multichannel images for prediction (channel1 = _0000, channel2 = _0001, ...), enabled by default

    [Postprocessing]
    - nnu.restore.orient: whether to restore the original orientation of the image in the predicted segmentation
    - nnu.restore.meta: whether to add the original metainformation of the image in the predicted segmentation
    - nnu.result.annotate: whether to add annotation metainformation in the predicted segmentation
    - nnu.result.colors: colors to assign to labels in the predicted segmentation
    - nnu.result.ext: file extension to use for the result prediction
    """

    def __init__(self, param: dict):
        self.task_name = None
        self.silent = False
        self.version = dict_get(param, 'nnu.version', default=2, dtype=int)
        self.task_id = dict_get(param, 'nnu.task', default=None, dtype=int)
        self.folds = dict_get(param, 'nnu.folds', default=None, dtype=List[int])
        self.plans = dict_get(param, 'nnu.plans', default='nnUNetPlans', dtype=str)
        self.configuration = dict_get(param, 'nnu.configuration', default='3d_fullres', dtype=str)
        self.verbose = dict_get(param, 'nnu.verbose', default=True, dtype=bool)

        self.trainer = dict_get(param, 'nnu.trainer', default='nnUNetTrainer', dtype=str)
        self._custom_trainer_file = dict_get(param, 'nnu.trainer.file', default=None, dtype=str)
        self._custom_trainer_name = dict_get(param, 'nnu.trainer.name', default=self.trainer, dtype=str)
        self._custom_trainer_uid = dict_get(param, 'nnu.trainer.uid', default=self.trainer, dtype=str)
        if self._custom_trainer_file is None and self._custom_trainer_uid != self.trainer:
            warn("Trainer uids (nnu.trainer.uids) are only allowed for custom trainers: ignoring the value {}.".format(
                self._custom_trainer_uid))
            self._custom_trainer_uid = self.trainer

        self.processes = dict_get(param, 'nnu.processes.general', default=2, dtype=int)
        self._processes_preprocess = dict_get(param, 'nnu.processes.preprocessing', default=self.processes, dtype=int)
        self._processes_export = dict_get(param, 'nnu.processes.export', default=self.processes, dtype=int)
        self._preprocessed_name = dict_get(param, 'nnu.preprocessed.name', default='_preprocessed')
        self._preprocessed_path = dict_get(param, 'nnu.preprocessed.path', default=None)

        self._predict_orient = dict_get(param, 'nnu.predict.orient', default=None, dtype=str)
        self._predict_checkpoint = dict_get(param, 'nnu.predict.checkpoint', default='final', dtype=str)
        self._predict_probs = dict_get(param, 'nnu.predict.probabilities', default=False, dtype=bool)
        self._predict_augment = dict_get(param, 'nnu.predict.augment', default=True, dtype=bool)
        self._predict_stepsize = dict_get(param, 'nnu.predict.stepsize', default=None, dtype=float)
        self._predict_cpu = dict_get(param, 'nnu.predict.cpu', default=False, dtype=bool)
        self._predict_split_mc = dict_get(param, 'nnu.predict.multichannel.split', default=True, dtype=bool)

        self._result_annotate = dict_get(param, 'nnu.result.annotate', default=True, dtype=bool)
        self._restore_meta = dict_get(param, 'nnu.restore.meta', default=True, dtype=bool)
        self._restore_orient = dict_get(param, 'nnu.restore.orient', default=True, dtype=bool)
        self._result_colors = dict_get(param, 'nnu.result.colors', default='ts2d')
        self._result_ext = dict_get(param, 'nnu.result.ext', default='seg.nrrd', dtype=str)

        self._config = None
        self._adapter = self._create_adaptee(self, self.version)

    @staticmethod
    def _create_adaptee(wrapper, version):
        adapteeType = {
            2: _NNUAdapterV2
        }.get(version)
        if adapteeType is None:
            raise PipelineError("No implementation for specified nnu version: {}".format(version))
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
                raise RuntimeError("CUDA is not available in the installed Pytorch package!")
        except:
            warn(
                "ERROR: The Pytorch package (pytorch/torch) is not correctly installed in the current python environment.\n"
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

    def get_env(self):
        if not self.silent:
            log()
            log("Running nnu-net with the following environment variables: ")
        env_set = dict()
        for dkey, skey in self._adapter.get_env_mapping().items():
            if skey in self._config:
                value = self._config.get(skey)
                env_set[dkey] = value
                if not self.silent:
                    log("'{}': '{}'".format(dkey, value))
        env = os.environ.copy()
        env.update(env_set)
        return env

    def configure(self, result_dir=None, data_dir=None, make_task=False, load_splits=True, override=False):
        nnu_config = dict()

        target_task_id = None
        target_task_name = None
        target_splits_path = None

        data_config_name = 'dataset.json'
        data_splits_name = self._adapter.get_splits_file_name()
        data_meta_name = 'meta.csv'

        if data_dir is not None:
            if not os.path.exists(data_dir):
                raise PipelineError("Invalid data directory: {}".format(data_dir))

            if make_task:
                target_task_id = self.task_id
                if self.version == 1:
                    data_task_name = 'Task{:03d}_{}'.format(target_task_id, self.task_name)
                elif self.version == 2:
                    data_task_name = 'Dataset{:03d}_{}'.format(target_task_id, self.task_name)
                else:
                    raise RuntimeError("Invalid nnU-Net Version: {}".format(self.version))
                data_task_dir = os.path.join(data_dir, data_task_name)
                mkdirs(data_task_dir)
            else:
                detected_tasks = nnu_find_datasets(data_dir, version=self.version)
                target_task_id, data_task_name = self._check_detected_tasks(detected_tasks, expected=self.task_id)
                data_task_dir = os.path.join(data_dir, data_task_name)

            data_config_path = os.path.join(data_task_dir, data_config_name)
            data_splits_path = os.path.join(data_task_dir, data_splits_name)
            data_meta_path = os.path.join(data_task_dir, data_meta_name)

            nnu_config.update({
                'data.dir': data_dir,
                'data.task.name': data_task_name,
                'data.task.dir': data_task_dir,
                'data.splits.path': data_splits_path,
                'data.meta.path': data_meta_path,
                'data.config.path': data_config_path,
            })
            target_task_name = data_task_name
            target_splits_path = data_splits_path
        else:
            detected_tasks = nnu_find_datasets(result_dir, version=self.version)
            target_task_id, target_task_name = self._check_detected_tasks(detected_tasks, expected=self.task_id)

        plans_name = '_'.join([self.plans, self.configuration])
        trainer_name = '__'.join([self.trainer, self.plans, self.configuration])

        if result_dir is not None:
            result_task_dir = os.path.join(result_dir, target_task_name)
            result_data_dir = os.path.join(result_task_dir, trainer_name)
            result_log_name = 'Log{}.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            result_log_path = os.path.join(result_task_dir, result_log_name)

            mkdirs(result_task_dir)

            result_config_name = data_config_name
            result_config_path = os.path.join(result_data_dir, result_config_name)
            result_config_data = dict()
            result_img_ext = None
            if os.path.exists(result_config_path):
                try:
                    result_config_data = read_json(result_config_path)
                    result_img_ext = result_config_data.get("file_ending")
                except:
                    warn("Failed to read the result dataset.json at: {}".format(result_config_path))

            result_fold_dirs = list()
            if os.path.exists(result_data_dir):
                result_fold_dirs.extend(os.path.join(result_data_dir, p)
                                        for p in os.listdir(result_data_dir)
                                        if re.match("fold_[0-9]+", p))
            result_fold_ids = list(int(os.path.basename(p).split('_')[1])
                                   for p in result_fold_dirs)

            preprocessed_dir = self._preprocessed_path
            if preprocessed_dir is None:
                preprocessed_name = default(self._preprocessed_name, '_preprocessed')
                preprocessed_dir = os.path.join(result_dir, preprocessed_name)
            else:
                preprocessed_name = os.path.basename(preprocessed_dir)

            preprocessed_task_dir = os.path.join(preprocessed_dir, target_task_name)
            preprocessed_data_dir = os.path.join(preprocessed_task_dir, plans_name)
            preprocessed_plans_path = os.path.join(preprocessed_task_dir, 'nnUNetPlans.json')

            if load_splits:
                preprocessed_splits_path = os.path.join(preprocessed_task_dir, data_splits_name)
                if override or not os.path.exists(preprocessed_splits_path):
                    if target_splits_path is not None:
                        log("Replacing the split-file with the custom split")
                        mkdirs(preprocessed_task_dir)
                        shutil.copy(target_splits_path, preprocessed_splits_path)
                    else:
                        warn("No split-file available!")

                expected_fold_data = read_json(preprocessed_splits_path)
                expected_fold_items_names = dict((id, fold['val'])
                                                 for id, fold in enumerate(expected_fold_data))
                expected_fold_ids = sorted(expected_fold_items_names.keys())
                expected_fold_total = len(expected_fold_ids)
                expected_fold_dirs = list(os.path.join(result_data_dir, 'fold_{}'.format(f)) for f in expected_fold_ids)

                nnu_config.update({
                    'preprocessed.splits.path': preprocessed_splits_path,

                    'expected.fold.total': expected_fold_total,
                    'expected.fold.ids': expected_fold_ids,
                    'expected.fold.dirs': expected_fold_dirs,
                    'expected.fold.items.names': expected_fold_items_names,
                })

            nnu_config.update({
                'preprocessed.name': preprocessed_name,
                'preprocessed.dir': preprocessed_dir,
                'preprocessed.task.dir': preprocessed_task_dir,
                'preprocessed.data.dir': preprocessed_data_dir,
                'preprocessed.plans.path': preprocessed_plans_path,

                'result.dir': result_dir,
                'result.task.dir': result_task_dir,
                'result.data.dir': result_data_dir,
                'result.log.name': result_log_name,
                'result.log.path': result_log_path,
                'result.fold.ids': result_fold_ids,
                'result.fold.dirs': result_fold_dirs,
                'result.config.name': result_config_name,
                'result.config.path': result_config_path,
                'result.img.ext': result_img_ext
            })

        nnu_config.update({
            'target.task': target_task_id,

            'data.splits.name': data_splits_name,
            'data.config.name': data_config_name,
        })
        self._config = nnu_config

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
            raise PipelineError("The specified task was not found - no existing tasks identified!")
        if expected is None:
            if len(tasks) > 1:
                raise PipelineError(
                    "The task id is ambiguous as multiple tasks are available ({}): "
                    "specify 'nnu.task'".format(format_array(tasks)))
            expected = list(tasks.keys())[0]
        if expected not in tasks:
            raise PipelineError("The selected task is not available: {}".format(expected))
        return expected, tasks[expected]

    def _filter_detected_trainers(self, detected, trainer=None, plan=None, config=None):
        res = []
        for e in detected:
            if ((trainer is None or detected['trainer'] == trainer) and
                    (plan is None or detected['plan'] == plan) and
                    (config is None or detected['config'] == config)):
                res.append(e)

    def get_exts(self):
        """
        returns a list of supported image file extensions
        """
        return self._adapter.get_exts()

    def get_labels(self) -> dict:
        """
        returns a dictionary of labels configured with the model
        """
        dataset = self._read_dataset_config()
        return self._adapter.get_configured_labels(dataset)

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

    def get_splits_file_name(self):
        raise NotImplementedError()

    def get_configured_ext(self, config: dict):
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
            from importlib.metadata import distribution, PackageNotFoundError
            try:
                dist = distribution('nnunetv2')
                entries = {ep.name for ep in dist.entry_points if ep.group == 'console_scripts'}
            except PackageNotFoundError:
                raise RuntimeError("Failed to query the installed entry points of nnUNet!")

            required = ['nnUNetv2_train', 'nnUNetv2_predict', 'nnUNetv2_plan_and_preprocess']
            for r in required:
                if r not in entries:
                    raise RuntimeError(f"The required entry point '{r}' was not discovered in the nnUNet package!")
        except:
            warn("ERROR: The nnUNet package (nnunetv2) is not correctly installed in the current python environment.\n"
                 "--- INSTRUCTIONS ---\n"
                 "Try reinstalling the package with: pip install --force-reinstall nnunetv2\n"
                 "--------------------\n")
            raise

    def get_configured_ext(self, config: dict):
        return str(config['file_ending']).lstrip('.')

    def get_configured_labels(self, config: dict) -> dict:
        return dict(enumerate(config['labels'].keys()))

    def get_exts(self):
        return ['png', 'bmp', 'nii.gz', 'nrrd', 'mha', 'tif', 'tiff']

    def get_splits_file_name(self):
        return 'splits_final.json'

    def get_dims(self) -> int:
        return 2 if ('2d' in str(self.nnu.configuration).lower()) else 3

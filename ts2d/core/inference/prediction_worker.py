import os
import sys
import traceback
import uuid
from time import time, sleep
from typing import List

import torch.multiprocessing as mp
from multiprocessing import current_process
import psutil

from ts2d.core.util.log import log
from ts2d.core.util.meta import set_annotation_meta
from ts2d.core.util.temp import SafeTemporaryDirectory
from ts2d.core.util.util import parse_int


def _pool_initializer(**kwargs):
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    def _default_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, (KeyboardInterrupt, OSError)):
            return
        sys.stdout.flush()
        sleep(0.1)
        log("Uncaught exception: [{}] {}".format(exc_type.__name__, exc_value))
        if current_process().name == 'MainProcess':
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        return
    sys.excepthook = _default_hook
    for k in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        os.environ[k] = ''


class PredictTask:
    def __init__(self, filenames: str | List[str], ofile: str, id: uuid.UUID = None):
        filenames = [filenames] if isinstance(filenames, str) else list(filenames)
        fn0 = filenames[0]
        name = os.path.splitext(os.path.basename(fn0))[0].rsplit('_', 1)[0]
        self.id = uuid.uuid4() if id is None else id
        self.name: str = name
        self.filenames: List[str] = filenames

        if not os.path.splitext(ofile)[1]:
            # no extension means ofile is a directory
            ofile = os.path.join(ofile, f"{name}.nrrd")
        self.ofile: str = ofile

        self.save_probabilities: bool = False
        self.overwrite: bool = True

        self.success = False
        self.error = None
        self.done = False

        self.timestamps = dict()
        self.timestamps['start'] = time()

    def print(self):
        status = ("failed" if self.error is not None
                  else ("finished" if self.success
                        else ("aborted" if self.done else "pending")))


        times = "".join([f"\n\t * {k}: {v - self.timestamps['start']:.02f}"
                         for k, v in self.timestamps.items() if k != 'start'])

        print(f"Task {self.id}: {status}\n"
              f"- input: {self.name}\n"
              f"- output: {self.ofile}\n"
              f"- times: {times}")

def _create_dummy_task(predictor):
    try:
        patch_size = tuple(predictor.configuration_manager.patch_size)
        spacing = tuple(predictor.configuration_manager.spacing)
        channels = len(predictor.dataset_json.get('channel_names', 1))

        import SimpleITK as sitk
        temp_img = sitk.Image(tuple(patch_size), sitk.sitkInt16 if channels == 1 else sitk.sitkVectorInt16, channels)
        temp_img.SetSpacing(spacing)

        temp_p = SafeTemporaryDirectory()
        fp_in = os.path.join(temp_p.name, 'dummy_input.nrrd')
        fp_out = os.path.join(temp_p.name, 'dummy_output.nrrd')
        sitk.WriteImage(temp_img, fp_in, True)
        task = PredictTask([fp_in], fp_out)

        # store the temporary folder instance with the task
        task.temp = temp_p
    except:
        # it's better to return nothing (therefore have no dummy task) than to fail
        traceback.print_exc()
        task = None
    return task


def _run_worker(predictor,
                task_queue: mp.Queue,
                done: mp.Condition, abort: mp.Event, started: mp.Event,
                tasks: dict, registry: dict, param: dict):
    if ('nnUNetPredictor' not in type(predictor).__name__):
        if callable(predictor):
            # lazy predictor loading
            try:
                predictor = predictor()
            except Exception as ex:
                traceback.print_exc()
                raise RuntimeError(f"Lazy predictor loading failed: {ex}")
            if 'nnUNetPredictor' not in type(predictor).__name__:
                raise RuntimeError(f"The lazy loading function must return an instance of nnUNetPredictor, found '{type(predictor).__name__}'")
        else:
            raise RuntimeError("Predictor must be an instance of nnUNetPredictor or a callable that returns it.")


    temp_p = None
    curr_p = psutil.Process()
    try:
        registry[curr_p.pid] = {
            'created': curr_p.create_time(),
            'name': curr_p.name(),
            'parent': curr_p.ppid()
        }
        started.set()
        #print(f"A new worker started running ({curr_p.pid})")
        while not abort.is_set():
            try:
                task_id = task_queue.get()
            except EOFError:
                # the queue was closed...
                return
            if task_id is None or task_id == 'shutdown':
                # soft shutdown communicated through the queue
                break
            if task_id == 'startup':
                # create a dummy image to process
                task = _create_dummy_task(predictor)
            else:
                task = tasks.get(task_id, None)
            if task:
                try:
                    task.timestamps['get'] = time()
                    _run_predict(predictor, task, param)
                    task.success = True
                except Exception as e:
                    traceback.print_exc()
                    task.error = e
                except (KeyboardInterrupt, SystemExit):
                    # dont bother with communication
                    raise
                finally:
                    task.timestamps['done'] = time()
                    task.done = True
                    if task_id == 'startup':
                        try:
                            task.temp.cleanup()
                            task.temp = None
                        except:
                            # not critical if the cleanup fails
                            pass
                    with done:
                        # update the task result
                        tasks[task_id] = task
                        done.notify()

            else:
                print("A submitted task id was not found by the worker...", file=sys.stderr)
    except (KeyboardInterrupt, SystemExit) as ex:
        print(f"Worker {curr_p.pid} was killed: ({type(ex).__name__})")
    except Exception as ex:
        print(f"Worker {curr_p.pid} crashed: ({type(ex).__name__})")
        traceback.print_exc()
    finally:
        registry.pop(curr_p, None)

def _run_predict(predictor, task: PredictTask, param: dict):
    try:
        ofile_truncated, ofile_ext = os.path.splitext(task.ofile)
        expected_ext = predictor.dataset_json['file_ending']
        if expected_ext != ofile_ext:
            raise RuntimeError(f"Expected '{expected_ext}' file extension for ofile, found: {ofile_ext}")
    except Exception as ex:
        raise RuntimeError(f"Failed parsing the output file name: {ex}") from ex

    try:
        odir = os.path.dirname(task.ofile)
        if not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)
    except Exception as ex:
        raise RuntimeError(f"Could not create output directory: {ex}") from ex

    try:
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
        data, _, data_properties = preprocessor.run_case(task.filenames,
                                                         None,
                                                         predictor.plans_manager,
                                                         predictor.configuration_manager,
                                                         predictor.dataset_json)
        task.timestamps['preprocessed'] = time()
    except Exception as ex:
        raise RuntimeError(f"Preprocessing failed for {task.name}: {ex}") from ex

    try:
        import torch
        data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        if predictor.device.type == 'cuda':
            data.pin_memory()
        prediction = predictor.predict_logits_from_preprocessed_data(data).cpu()
        task.timestamps['predicted'] = time()
    except Exception as ex:
        raise RuntimeError(f"Prediction failed for {task.name}: {ex}") from ex

    try:
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        export_prediction_from_logits(prediction, data_properties,
                                      predictor.configuration_manager,
                                      predictor.plans_manager,
                                      predictor.dataset_json,
                                      ofile_truncated,
                                      task.save_probabilities)
        task.timestamps['exported'] = time()
    except Exception as ex:
        raise RuntimeError(f"Export failed for {task.name}: {ex}") from ex

    try:
        labels = param.get('labels')
        colors = param.get('colors')
        # In case param is empty, predictions should default to nnu's dataset.json labels
        labels = labels if isinstance(labels, dict) else predictor.dataset_json.get('labels')
        colors = colors if isinstance(colors, dict) else None
        labels = {parse_int(k): v for k, v in labels.items()}
        if labels or colors:
            import SimpleITK as sitk
            reader = sitk.ImageFileReader()
            reader.SetFileName(task.ofile)
            reader.ReadImageInformation()
            img = reader.Execute()
            set_annotation_meta(img, labels, colors)
            sitk.WriteImage(img, task.ofile, True)
    except Exception as ex:
        raise RuntimeError(f"Postprocessing failed for {task.name}: {ex}") from ex

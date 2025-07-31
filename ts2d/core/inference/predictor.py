import multiprocessing
import os
import sys
import uuid
from time import time, sleep
from types import NoneType
from typing import List, Dict

import psutil
import torch.multiprocessing as mp
from multiprocessing.pool import Pool

from ts2d.core.inference.prediction_worker import _run_worker, PredictTask, _pool_initializer

class ParallelPredictor:
    def __init__(self):
        self._workers: Dict[int, dict] = dict()
        self._task_queue: mp.Queue = None
        self._task_done: mp.Condition = None
        self._task_abort = mp.Event = None
        self._task_started: mp.Event = None
        self._tasks: dict = None

        self._param: dict = dict()

    @property
    def labels(self):
        return self._param.get('labels')

    @labels.setter
    def labels(self, value: dict):
        self._param['labels'] = value

    @property
    def colors(self):
        return self._param.get('colors')

    @colors.setter
    def colors(self, value: dict):
        self._param['colors'] = value

    def start(self, predictor: 'nnUNetPredictor',
              num_workers: int = 4, pool: bool | Pool = True, daemon=True):
        manager = mp.Manager()
        self._task_queue = manager.Queue()
        self._task_done = manager.Condition()
        self._task_abort = manager.Event()
        self._task_started = manager.Event()
        self._tasks = manager.dict()
        self._workers = manager.dict()

        param = manager.dict()
        param.update(self._param)
        self._param = param
        kwargs = {
            'predictor': predictor,
            'task_queue': self._task_queue,
            'done': self._task_done,
            'abort': self._task_abort,
            'started': self._task_started,
            'tasks': self._tasks,
            'registry': self._workers,
            'param': self._param
        }
        if os.name != 'nt':
            multiprocessing.set_start_method('spawn', force=True)
        if pool:
            # pools are faster to initialize than individual processes
            if isinstance(pool, bool):
                # since we do not keep a reference to the pool and call terminate on it,
                # the pool may through an OSError exception on its deletion, although there is no real harm in that
                # however, we should suppress this exception
                def _suppress_delete_exception(self):
                    try:
                        self.__del__()
                    except OSError:
                        pass

                pool = mp.Pool(processes=num_workers, initializer=_pool_initializer)
                pool.__del__ = _suppress_delete_exception
            else:
                # pool was provided by the user
                pass

            for _ in range(num_workers):
                pool.apply_async(_run_worker, kwds=kwargs)

        else:
            assert num_workers > 0
            for rank in range(num_workers):
                p = mp.Process(target=_run_worker, kwargs=kwargs,
                               name=f"PredictionWorker#{rank}",
                               daemon=daemon)
                p.start()
        for _ in range(num_workers):
            self._task_queue.put('startup')


    def predict(self,
                filenames: str | List[str],
                ofile: str,
                id: uuid.UUID = None,
                save_probabilities: bool = False,
                overwrite: bool = True,
                wait=True):

        if self._task_queue is None:
            raise RuntimeError("Predictor not started. Call start() first.")

        task = PredictTask(filenames, ofile, id=id)
        task.save_probabilities = save_probabilities
        task.overwrite = overwrite
        self._tasks[task.id] = task
        self._task_queue.put(task.id)

        if wait:
            return self.wait(task.id)
        return task.id

    def wait(self, ids: str | uuid.UUID | PredictTask | List[uuid.UUID] | List[PredictTask] | NoneType = None,
             timeout=None,
             fail=True):
        if ids is None:
            ids =  list(self._tasks.keys())
        multiple = not isinstance(ids, (str, uuid.UUID, PredictTask))
        wait_ids, results = (ids, dict()) if multiple else ([ids], None)
        wait_ids = set(t.id if isinstance(t, PredictTask) else t for t in wait_ids)
        n_ids = len(wait_ids)
        start = time()

        # while there is anything to wait for
        while wait_ids:
            # is the task finished already?
            for id in list(wait_ids):
                task = self._tasks.get(id, None)
                if task is None:
                    # let's look if a task object exists, but note: named tasks are treated differently
                    if not isinstance(id, str):
                        # task does no longer exist...
                        wait_ids.remove(id)
                elif task.done:
                    # task ended (successfully or not)
                    if multiple:
                        wait_ids.remove(id)
                        results[id] = task
                    else:
                        return task

            # check if workers are still running
            workers_stopped = self._wait_stop(timeout=0)

            # check if the timeout was reached
            timeout_reached = timeout is not None and time() - start > timeout

            if workers_stopped or timeout_reached:
                if fail and wait_ids:
                    raise RuntimeError(f"Failed to wait for {len(wait_ids)} of {n_ids} tasks!")
                return results
            with self._task_done:
                # wait for the next task to finish
                self._task_done.wait(timeout=0.10)
        return results

    def stop(self, timeout=10):
        # soft shutdown
        try:
            try:
                self._task_abort.set()
            except FileNotFoundError:
                # event may have been released already
                pass
            for pid, info in self._fetch_workers().items():
                try:
                    self._task_queue.put(None)
                except FileNotFoundError:
                    # queue may have been released already
                    pass
        except FileNotFoundError:
            pass

        if not self._wait_stop(timeout=timeout):
            print("Interrupting the workers did not work, terminating the processes instead...", file=sys.stderr)
            workers = self._get_workers()
            for worker in workers:
                try:
                    worker.terminate()
                except psutil.NoSuchProcess:
                    pass # dead already

    def _wait_stop(self, timeout: float):
        start = time()
        try:
            while True:
                if self._task_started.is_set():
                    workers = self._get_workers()
                    if not workers:
                        break
                if time() - start > timeout:
                    return False
                sleep(0.01)
            return True
        except:
            # we can not communicate with the workers anymore
            # likely the server has been stopped already
            return True

    def _get_workers(self):
        alive = []
        for pid, info in self._fetch_workers().items():
            try:
                p = psutil.Process(pid)
                if (
                        p.ppid() != info.get('parent')
                        or p.create_time() != info.get('created')
                        or p.name() != info.get('name')
                ):
                    raise RuntimeError("Mismatch")
                if not p.is_running():
                    raise RuntimeError("Stopped")
                alive.append(p)
            except:
                # may be triggered Mismatch/Stopped or psutil failing to access the process
                self._workers.pop(pid, None)
        return alive

    def _fetch_workers(self):
        # Manager.dict behaves weired after the Manager dies...
        workers = self._workers.items()
        return dict(workers) if hasattr(workers, '__iter__') else dict()

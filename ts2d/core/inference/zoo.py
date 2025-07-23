import os
import sys
from typing import Optional

from ts2d.core.inference.database import URLDataBase, FileDataBase, DataBase
from ts2d.core.inference.nnu import NNUModel, NNUProcessModel, _describe_model
from ts2d.core.util.config import get_shared_urls
from ts2d.core.util.file import read_json
from ts2d.core.util.log import log
from ts2d.core.util.path import get_local_models_root
from ts2d.core.util.types import dict_merge


class NNUZoo:
    """
    Represents the interface to the remote and local nnu model zoo.
    The class automatically copies content to the local database (defaults to the AppData directory) when
    an nnu model is accessed.
    """

    def __init__(self, remote: Optional[DataBase] = None, local: Optional[str | DataBase] = None):
        """
        Instantiate a new NNUZoo object
        :param remote: an optional custom instance for the remote database, if None the default URL database will be used, if False, no remote will be configured
        :param local: the path to the local root for pretrained models, if None the default path will be used, cannot be disabled
        """

        if local is None:
            local = get_local_models_root()
        if remote is None:
            remote = URLDataBase(get_shared_urls())
        if remote == False:
            remote = None
        self._remote = remote
        self._local = local if isinstance(local, DataBase) else (FileDataBase(local, readonly=False) if isinstance(local, str) else None)
        assert self._local is not None, "A valid local database instance must be configured for NNU zoo"

    @property
    def remote(self):
        """
        The remote database, i.e., the location to download atlases from
        """
        return self._remote

    @property
    def local(self):
        """
        The local database, i.e., the location to cache atlas data at
        """
        return self._local

    def access(self, model, revision=None) -> dict:
        """
        access the specified pretrained model, this involves checking the local database - when the specified atlas does not exist
        it will be automatically be copied from the remote.
        """
        if self.remote:
            if revision is None:
                # not fully specified: look for a later model on the remote
                try:
                    res = self.remote.has(model, revision)
                    if res:
                        model, revision = res
                except Exception as ex:
                    raise RuntimeError("Failed to check the latest model revision on the remote dataset: {}".format(ex))

        desc = _describe_model(model=model, revision=revision)
        res = self.local.has(model, revision)
        if res:
            model, revision = res
        elif self.remote:
            res = self.remote.has(model, revision)
            if not res:
                raise RuntimeError("No pretrained model '{}' in remote or local database!".format(desc))

            model, revision = res
            msg = "Copying pretrained model '{}' from remote to local database".format(desc)
            try:
                log("{}...".format(msg), end='')
                self.remote.copy(self.local.root, model=model, revision=revision)
                log("\r{} - DONE".format(msg))
            except:
                log("\r{} - FAILED".format(msg), file=sys.stderr)
                raise
            if not self.local.has(model, revision):
                raise RuntimeError("Model '{}' is not available in local dataset after copying!")
        else:
            raise RuntimeError("No pretrained model '{}' in the local database!".format(desc))

        return {
            'root': self.local._access_resource_paths(model, revision, fail=True)[0],
            'model': model,
            'revision': revision
        }

    def load(self, name, interface: str='cmd', param: Optional[dict] = None, **kwargs):
        """
        Loads a pretrained model and returns a NNUModel instance
        Different derived classes can be returned depending on the interface used to access the model:
        - 'cmd' or 'commandline': NNUCommandlineModel, uses the commandline interface to predict results
        - 'svc' or 'server': NNUServerModel, hosts the model in python processes and uses a server interface to predict results
        :param name: name of the model to load
        :param interface: (optional) type of interface used to access to the model, defaults to 'cmd' for commandline, options: 'cmd', 'svc'
        :param param: additional parameters used to configure the model, overrides config['param']
        :param kwargs: further arguments used to access the model: revision
        """
        try:
            config = self.access(name, **kwargs)
            root = config['root']
            if not os.path.exists(root):
                raise RuntimeError("Failed to locate the root for the model: {}".format(
                    _describe_model(model=name, **kwargs)))
            try:
                jpath = os.path.join(config['root'], 'model.json')
                config.update(read_json(jpath, warn=False))
            except Exception as ex:
                raise RuntimeError("Failed to load a model configuration: {}".format(ex))
            return NNUZoo._create_model(interface, config, param if param is not None else dict())
        except Exception as ex:
            raise RuntimeError("Failed to load a pretrained model: {}".format(ex))

    def clear(self, model: Optional[str]=None, revision: Optional[int]=None):
        self.local.clear(model=model, revision=revision)

    @staticmethod
    def _create_model(interface: str, config: dict, param: dict) -> 'NNUModel':
        config['param'] = dict_merge(config['param'], param)
        interface = interface.lower()
        if interface in { 'prc', 'process' }:
            return NNUProcessModel(config)
        else:
            raise RuntimeError("Invalid model type: {}".format(type))

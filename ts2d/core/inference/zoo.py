import os
import sys
from typing import Optional, List

from ts2d.core.inference.database import URLDataBase, FileDataBase, DataBase
from ts2d.core.inference.nnu import NNUModel, NNUProcessModel, _describe_model
from ts2d.core.util.config import get_shared_urls, get_model_resolve_map
from ts2d.core.util.file import read_json
from ts2d.core.util.log import log
from ts2d.core.util.path import get_local_models_root
from ts2d.core.util.types import dict_merge, default, unwrap_singular


class NNUZoo:
    """
    Represents the interface to the remote and local nnu model zoo.
    The class automatically copies content to the local database (defaults to the AppData directory) when
    an nnu model is accessed.
    """

    def __init__(self,
                 remote: Optional[DataBase] = None,
                 local: Optional[str | DataBase] = None):
        """
        Instantiate a new NNUZoo object
        :param remote: an optional custom instance for the remote database, if None the default URL database will be used, if False, no remote will be configured
        :param local: the path to the local root for pretrained models, if None the default path will be used, cannot be disabled
        """

        if local is None:
            local = get_local_models_root()
        if remote == False:
            remote = None
        elif remote is None:
            remote = URLDataBase(get_shared_urls())
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

    def resolve(self, key: str, unique_model: bool=False) -> List[str]:
        """
        For a key, returns a list of matching model ids.
        If the remote database is configured, it will check the remote for matching models, otherwise the local repository is used.
        :param key: the key to resolve
        :param unique_model: if True, only models belonging to the same base model are returned (i.e., models for subgroups of the same configuration)
        otherwise any matching models are returned). If multiple base models are resolved with this key, the first one is selected.
        :return: a list of model ids that match the key
        """
        # resolve default keys
        map = get_model_resolve_map()
        while key in map:
            key = map[key]

        db = default(self.remote, self.local)
        if unique_model:
            # take the first unique model for the key
            models = sorted(db.models(key=key))
            if not models:
                raise RuntimeError(f"No models resolved for key '{key}'")
            model = models[0]
            return db.ids(model=model)
        return db.ids(key=key)


    def access(self, id: str, revision: int | str | None =None) -> dict:
        """
        access the specified pretrained model, this involves checking the local database - when the specified atlas does not exist
        it will be automatically be copied from the remote
        :param id: the id of the model to access, do not use ambiguous keys
        :param revision: (optional) the revision of the model to access, if None, the latest revision will be used
        """

        ids = self.resolve(id)
        if len(ids) > 1:
            raise RuntimeError(f"The model id '{id}' is ambiguous (matches {', '.join(ids)})")
        if self.remote:
            if revision is None:
                # not fully specified: look for a later model on the remote
                try:
                    res = self.remote.has(key=id, revision=revision)
                    if res:
                        revision = self.remote.latest(key=id)
                except Exception as ex:
                    raise RuntimeError("Failed to check the latest model revision on the remote dataset: {}".format(ex))

        desc = _describe_model(key=id, revision=revision)
        res = self.local.has(key=id, revision=revision)
        if res:
            if revision is None:
                revision = self.local.latest(key=id)
        elif self.remote:
            res = self.remote.has(key=id, revision=revision)
            if not res:
                raise RuntimeError("No pretrained model '{}' in remote or local database!".format(desc))
            if revision is None:
                revision = self.remote.latest(key=id)
            msg = "Copying pretrained model '{}' from remote to local database".format(desc)
            try:
                log("{}...".format(msg), end='')
                self.remote.copy(self.local.root, key=id, revision=revision)
                log("\r{} - DONE".format(msg))
            except:
                log("\r{} - FAILED".format(msg), file=sys.stderr)
                raise
            if not self.local.has(key=id, revision=revision):
                raise RuntimeError("Model '{}' is not available in local dataset after copying!")
        else:
            raise RuntimeError("No pretrained model '{}' in the local database!".format(desc))

        # fetch the local model information
        info = self.local.get(key=id, revision=revision)
        info['root'] = unwrap_singular(self.local._access_resource_paths(key=info['id'], revision=revision, fail=True))
        return info

    def load(self, id: str, interface: str= 'cmd', param: Optional[dict] = None, **kwargs):
        """
        Loads a pretrained model and returns a NNUModel instance
        Different derived classes can be returned depending on the interface used to access the model:
        - 'cmd' or 'commandline': NNUCommandlineModel, uses the commandline interface to predict results
        - 'svc' or 'server': NNUServerModel, hosts the model in python processes and uses a server interface to predict results
        :param id: the id of the model to load
        :param interface: (optional) type of interface used to access to the model, defaults to 'cmd' for commandline, options: 'cmd', 'svc'
        :param param: additional parameters used to configure the model, overrides config['param']
        :param kwargs: further arguments used to access the model: revision
        """
        try:
            config = self.access(id=id, **kwargs)
            root = config['root']
            if not os.path.exists(root):
                raise RuntimeError("Failed to locate the root for the model: {}".format(
                    _describe_model(key=id, **kwargs)))
            try:
                jpath = os.path.join(config['root'], 'model.json')
                config.update(read_json(jpath))
            except Exception as ex:
                raise RuntimeError("Failed to load a model configuration: {}".format(ex))
            return NNUZoo._create_model(interface, config, param if param is not None else dict())
        except Exception as ex:
            raise RuntimeError("Failed to load a pretrained model: {}".format(ex))

    def clear(self, key: Optional[str]=None, revision: Optional[int]=None):
        self.local.clear(key=key, revision=revision)

    @staticmethod
    def _create_model(interface: str, config: dict, param: dict) -> 'NNUModel':
        config['param'] = dict_merge(config['param'], param)
        interface = interface.lower()
        if interface in { 'prc', 'process' }:
            return NNUProcessModel(config)
        else:
            raise RuntimeError("Invalid model type: {}".format(type))

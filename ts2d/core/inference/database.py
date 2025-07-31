import os
import sys
import json
import shutil
import tempfile
import zipfile
from glob import glob
from typing import Optional
import dload
import gdown

from ts2d.core.util.log import warn, log
from ts2d.core.util.temp import SafeTemporaryDirectory
from ts2d.core.util.types import default, as_set, dict_merge
from ts2d.core.util.util import parse_int, removeprefix, rmdirs, removeall, isemptydir, format_array


def decompose_model_key(key: str):
    """
    returns the model and group from a model key
    """
    model, group = key.rsplit('_', maxsplit=1) if '_' in key else (key, None)
    return model, group

def _describe_model(**kwargs):
    key = kwargs.get('key')
    if key is not None:
        model, group = decompose_model_key(key)
    else:
        model = kwargs['model']
        group = kwargs.get('group')
    revision = kwargs.get('revision')
    folds = kwargs.get('folds')
    return ''.join(([f"{model}"]
                    + ([] if group is None else [f" for {group}"])
                    + ([] if revision is None else [f" at {_revision_str(revision)}"]))
                    + ([] if folds is None else [f"(folds: {', '.join(str(f) for f in folds)}"])
                    + ([] if key is None else [f"(key: {key})"]))

def _revision_str(revision: int):
    return 'r{:03d}'.format(revision) if isinstance(revision, int) else revision

def _parse_revision(rn: str):
    return parse_int(rn if isinstance(rn, int) else removeprefix(str(rn), 'r'))


class DataBase:
    def copy(self, dest_root, key: str, revision: Optional[int] = None):
        raise NotImplementedError()

    def has(self, model: str | None = None, group: str | None = None, key: str | None = None, revision: int | None = None):
        return bool(self.list(model=model, group=group, key=key, revision=revision))

    def ids(self, model: str | None = None, group: str | None = None, key: str | None = None, revision: int | None = None):
        return sorted(set(f'{m}_{g}' for (m, g, r), p in self.list(model=model, group=group, key=key, revision=revision).items()))

    def get(self, model: str | None = None, group: str | None = None, key: str | None = None, revision: int | None = None) -> dict:
        """
        returns the details for the first model found in the database that matches the specified parameters
        """
        id, (m, g, r, p) = sorted(dict((f'{m}_{g}', (m, g, r, p)) for (m, g, r), p in self.list(model=model, group=group, key=key, revision=revision).items()).items(), key=lambda t: t[0])[0]
        return {
            'id': id,
            'model': m,
            'group': g,
            'revision': r,
            'path': p
        }

    def models(self, group: str | None = None, revision: int | None = None, key: str | None = None):
        return sorted(set(m for (m, g, r), p in self.list(group=group, revision=revision, key=key).items()))

    def groups(self, model: str | None = None, revision: int | None = None, key: str | None = None):
        return sorted(set(g for (m, g, r), p in self.list(model=model, revision=revision, key=key).items()))

    def revisions(self, model: str | None = None, group: str | None = None, key: str | None = None) -> list:
        return sorted(set(r for (m, g, r), p in self.list(model=model, group=group, key=key).items()))

    def latest(self, model: str | None = None, group: str | None = None, key: str | None = None) -> Optional[int]:
        revs = self.revisions(model=model, group=group, key=key)
        if not revs:
            return None
        return revs[-1]

    def _enumerate(self):
        raise NotImplementedError()

    def _match_model_str(self, match: str | None, model: str):
        if match is None:
            return True
        if '-' in model:
            match = match.split('-')
            model = model.split('-')
            for i in range(len(model)):
                if i < len(match) and match[i] and match[i] != model[i]:
                    return False
            return True

        return model == match

    def list(self, model: str | None = None, group: str | None = None, key: str | None = None,  revision: str | int | None = None):
        if key is not None:
            model, group = decompose_model_key(key)
        revision = _parse_revision(revision) if isinstance(revision, str) else revision
        res = dict()
        for _model, _group, _revision, _path in self._enumerate():
            if (self._match_model_str(model, _model)
                and (revision is None or revision == _revision)
                and (group is None or group == _group)):
                res[(_model, _group, _revision)] = _path
        return res


class FileDataBase(DataBase):
    def __init__(self, root: str, readonly: bool = True):
        super().__init__()
        self._root = root
        self._readonly = readonly

    @property
    def root(self):
        return self._root

    @property
    def readonly(self):
        return self._readonly

    def _enumerate(self):
        for dn in glob(os.path.join(self._root, '*', 'r*')):
            rdn = os.path.relpath(dn, self._root)
            try:
                model, rn = os.path.split(rdn)
                rn = _parse_revision(rn)
                if rn is None:
                    raise RuntimeError("Failed to parse a revision from {}".format(rn))

                model, group = model.rsplit('_', maxsplit=1) if '_' in model else (model, None)
                if group is None:
                    raise RuntimeError("Failed to parse a structure group from {}".format(model))

                yield model, group, rn, dn
            except Exception as ex:
                warn("Failed to list model from database folder: {} ({})".format(rdn, ex))


    def clear(self, key: str, revision: Optional[int] = None):
        if self.readonly:
            raise RuntimeError("Clear is not allowed for readonly Database!")
        paths = self._access_resource_paths(key=key, revision=revision, fail=False, minimal=True)
        for fp in paths:
            if os.path.exists(fp):
                if os.path.isdir(fp):
                    rmdirs(fp)
                else:
                    removeall(fp)
        paths = as_set(self._access_resource_paths(key=key, fail=False)).union(
            self._access_resource_paths(key=key, revision=revision, fail=False))
        for fp in paths:
            if isemptydir(fp):
                rmdirs(fp)

    def copy(self, dest_root, key: str, revision: Optional[int] = None):
        paths = self._access_resource_paths(key=key, revision=revision, fail=True)
        for fp in paths:
            rp = os.path.relpath(fp, self.root)
            dst = os.path.join(dest_root, rp)
            if os.path.isdir(fp):
                os.makedirs(dst, exist_ok=True)
                shutil.copytree(fp, dst, dirs_exist_ok=True)
            elif os.path.isfile(fp):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(fp, dst)
            else:
                raise RuntimeError("Unknown resource type for path: {}".format(fp))

    def _access_resource_paths(self, key: Optional[str] = None, revision: Optional[int] = None,
                               fail=True):
        path = self._root
        if not os.path.exists(path):
            raise RuntimeError("The database root does not exist: {}".format(path))
        if key is not None:
            desc = _describe_model(key=key, revision=revision)
            key = str(key).lower().strip()
            path = os.path.join(path, key)
            if not os.path.exists(path):
                if fail:
                    raise RuntimeError("The model does not exist in database: {}".format(desc))
                else:
                    return []

            if revision is not None:
                revision_str = _revision_str(revision)
                path = os.path.join(path, revision_str)
                if not os.path.exists(path):
                    if fail:
                        raise RuntimeError(f"Revision {revision_str} does not exist for model {key} in database")
                    else:
                        return []
        return [path]


class URLDataBase(DataBase):
    def __init__(self, urls: dict):
        super().__init__()
        self._urls = urls

    def copy(self, dest_root, key: str, revision: Optional[int] = None):
        for (m, g, rn), url in self.list(key=key, revision=revision).items():
            subkey = f'{m}_{g}-{_revision_str(rn)}'

            # download the zip to a temporary folder and extract to the destination
            with SafeTemporaryDirectory() as temp:
                temp_zip = os.path.join(temp, f'{subkey}.zip')
                gdown.download(url, output=temp_zip, quiet=False, fuzzy=True)
                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                    zip_ref.extractall(dest_root)

    def _enumerate(self):
        for model, mval in self._urls.items():
            for revision, rval in mval.items():
                for group, url in rval.items():
                    rn = _parse_revision(revision)
                    yield model, group, rn, url

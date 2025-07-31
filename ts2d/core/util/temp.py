import os
import shutil
import tempfile
import time
from tempfile import TemporaryDirectory

import psutil

from ts2d.core.util.file import read_json, write_json, enumerate_files
from ts2d.core.util.log import warn
from ts2d.core.util.util import mkdirs


class SafeTemporaryDirectory(TemporaryDirectory):
    """
    An overload of TemporaryDirectory that does not fail when the directory is not able to cleanup.
    Per default, a temporary directory is created in subdirectory 'ts2d' in the system default temporary location.
    Temporary directories are stored with a ~INFO.json file that identifies the process owning the directory, on creation
    of a new temporary directory, existing ones are checked and deleted if their owning processes are no longer existant.
    After 3 failed retries cleanup is aborted without raising an Exception.
    """

    def __init__(self, drive=None, **kwargs):

        dn = kwargs.get('dir')
        if dn is None:
            # TS2D_TEMP can overrule the default temporary directory if the directory was not specified with 'dir'
            dn = os.environ.get('TS2D_TEMP')
            if dn is None:
                if drive:
                    dir = os.path.join(drive, os.sep, '~temp')
                else:
                    dir = tempfile.gettempdir()
                dn = os.path.join(dir, 'ts2d')
            kwargs['dir'] = dn
            try:
                mkdirs(dn)
            except Exception as ex:
                warn("The preferred destination for temporary data is not available: {} ({})".format(dn, ex))
                dn = None
        super().__init__(**kwargs)
        if dn is not None:
            prefix = kwargs.get('prefix')
            suffix = kwargs.get('suffix')
            TemporaryDestination._clear_old(dn, prefix=prefix, suffix=suffix)
        self._write_info()

    def _write_info(self):
        fn = os.path.join(self.name, '~INFO.json')
        p = psutil.Process(os.getpid())
        info = {
            'pid': p.pid,
            'time': p.create_time()
        }
        write_json(info, fn)

    @staticmethod
    def _clear_old(root, prefix=None, suffix=None):
        if os.path.exists(root):
            for dn in os.listdir(root):
                if (prefix is None or dn.endswith(prefix)) and (suffix is None or dn.endswith(suffix)):
                    df = os.path.join(root, dn)
                    try:
                        if os.path.exists(df):
                            fn = os.path.join(root, dn, '~INFO.json')
                            do_clear = False
                            for retry in range(2):
                                try:
                                    info = read_json(fn) if os.path.exists(fn) else None
                                    break
                                except:
                                    # reading the info file failed, maybe we collided with the writing process... retry
                                    time.sleep(0.05)
                            else:
                                # the retry loop finished - the file is corrupted and should be cleared!
                                do_clear = True

                            if info and not do_clear:
                                # the info file was read, check the process
                                pid = info.get('pid')
                                ptime = info.get('time')
                                if not psutil.pid_exists(pid):
                                    do_clear = True
                                else:
                                    p = psutil.Process(pid)
                                    do_clear = p.create_time() != ptime
                            if do_clear:
                                for retry in range(3):
                                    try:
                                        if os.path.exists(df):
                                            shutil.rmtree(df)
                                        break
                                    except FileExistsError | FileNotFoundError:
                                        # someone else is deleting files...
                                        break
                                    except:
                                        # something went wrong, try again
                                        time.sleep(0.05)
                                else:
                                    warn("Failed removing an expired temporary folder: {}".format(df))
                    except:
                        pass

    def __exit__(self, *args, **kwargs):
        ignoreOnce = True
        do_warn = True
        for i in range(3):
            try:
                self.cleanup()
                break
            except NotADirectoryError as ex:
                # a curious error that seems to occur with the ~INFO.json file
                if ignoreOnce:
                    ignoreOnce = False
                    do_warn = False
                    i -= 1
            except Exception as ex:
                pass

            if do_warn:
                warn("Cleanup attempt {} of temporary directory failed ({})".format(i+1, self.name))

class TemporaryDestination(SafeTemporaryDirectory):
    """
    An overload of SafeTemporaryDirectory which allows the user to write files to a temporary directory
    the contents of which are moved to the destination directory on cleanup.
    Using TemporaryDestination, partial files as a result of program termination can be avoided
    :param dest_dir: The destination directory to mirror
    :param dest_dir: The destination file to mirror, only used to extract the parent directory which is then used like dest_dir
    """
    def __init__(self, dest_dir: str=None, dest_file=None, discard_on_error=True, optimize_drive=True, **kwargs):
        if dest_file:
            if dest_dir:
                raise RuntimeError("Either dest_dir or dest_file need to be specified, not both!")
            dest_dir = os.path.dirname(dest_file)
        if not dest_dir:
            raise RuntimeError("Either dest_dir or dest_file need to be specified!")

        if optimize_drive:
            kwargs['drive'] = os.path.splitdrive(dest_dir)[0]
        super().__init__(**kwargs)
        self.dest_dir = dest_dir
        self.discard_on_error = discard_on_error

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None or not self.discard_on_error:
            self.move()
        super().__exit__(exc_type, exc_value, exc_traceback)
        return self

    def __str__(self):
        return self.name

    def move(self):
        os.makedirs(self.dest_dir, exist_ok=True)
        for fp in enumerate_files(self.name):
            name = os.path.basename(fp)
            if not name.startswith('~'):
                rfp = os.path.relpath(fp, self.name)
                dfp = os.path.join(self.dest_dir, rfp)
                ddn = os.path.dirname(dfp)
                os.makedirs(ddn, exist_ok=True)

                if os.path.splitdrive(fp)[0] == os.path.splitdrive(dfp)[0]:
                    os.replace(fp, dfp)
                else:
                    shutil.move(fp, dfp) # -> slower but necessary between different drives
        try:
            super().cleanup()
        except:
            pass

    def remap(self, path: str):
        """
        maps a location pointing to the destination folder to the temporary directory
        """
        rfp = os.path.relpath(path, self.dest_dir)
        return os.path.join(self.name, rfp)

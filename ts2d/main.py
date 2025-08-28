import os
import shutil
from glob import glob

from ts2d.core.util.log import log, log_silent
from ts2d.tool import TS2D
from ts2d.core.util.config import get_default_model


def _enumerate_cases(src: str):
    isdir = os.path.isdir(src)
    src = glob(os.path.join(src, "*.*")) if isdir else [src]
    for fp in src:
        try:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Source file does not exist: {fp}")
            if not os.path.isfile(fp):
                raise ValueError(f"Source is not a regular file: {fp}")

            _, fn = os.path.split(fp)
            if not '.' in fn:
                raise ValueError(f"Source file does not have an extension: {fn}")

            name, ext = fn.split(".", maxsplit=1)
            if not ext in ('nrrd', 'nii', 'nii.gz', 'mha', 'mhd'):
                raise ValueError(f"Unsupported file extension: {ext} in {fn}")
            yield name, fp
        except:
            # when enumerating files, we can skip files that do not meet our criteria, otherwise we propagate the error
            if isdir:
                continue
            raise


def ts2d_run(src: str,
             dest: str,
             model: str = None,
             use_remote: bool = True,
             fetch_remote: bool = True,
             collapse: bool = False,
             visualize: bool = True,
             save_all: bool = False,
             silent: bool = False):
    """
    Streamlined function to run TS2D on one or more images.
    :param src: File path to the input image or directory of images.
    :param dest: Directory path where output results will be stored.
    :param model: Model key for prediction; uses the default if not specified.
    :param use_remote: If False, only locally available models are used; if True, models may be downloaded from a remote source.
    :param fetch_remote: If True, the model URLs are loaded from the remote shared.json file (from the main branch); if False, only the local clone is used.
    :param collapse: If True, collapses the projected images to 2D, otherwise the 3D geometry is preserved.
    :param visualize: If True, visualizes the final result as PNG images; if 'all', visualizes all results from every model.
    :param save_all: If True, saves results for each individual model; otherwise, only the final result is saved.
    """
    model = get_default_model() if model is None else model
    content = 'all' if visualize else 'file'
    models = 'all' if save_all else 'final'

    log_silent(silent)

    tsize = shutil.get_terminal_size(fallback=(120, 20))
    bar = '#' * tsize.columns
    log(f"\n{bar}\n"
        "TS2D is a research tool. It is NOT validated for clinical use and should NOT be used for medical diagnosis or treatment.\n"
        "Please cite the following paper when using TS2D:\n"
        "Sabrowsky-Hirsch, B., Alshenoudy, A., Thumfart, S., & Giretzlehner, M. (2025, July). "
        "TotalSegmentator 2D: A Tool for Rapid Anatomical Structure Analysis. "
        "In Annual Conference on Medical Image Understanding and Analysis (pp. 32-43). Cham: Springer Nature Switzerland.\n"
        f"{bar}\n")

    with TS2D(key=model, use_remote=use_remote, fetch_remote=fetch_remote) as model:
        cases = list(_enumerate_cases(src))
        n_cases = len(cases)
        log(f"Predicting {n_cases} case{'s' if n_cases != 1 else ''}")
        for case_id, (name, path) in enumerate(cases):
            log(f"[{case_id+1}/{n_cases}] Processing: {name}")
            res = model.predict(path, collapse=collapse)
            res.save(dest=dest, name=name, models=models, content=content, targets=['segmentation', 'projection'])

def ts2d_entry_point():
    """
    Entry point for the ts2d package.
    This function is called when the package is run as a script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Runs TotalSegmentator2D (TS2D) on images or directories of images to automatically segment anatomical structures.")
    parser.add_argument("--src", "-i", "--input", type=str, help="Input image file or directory. Supported formats are: nrrd, nii, nii.gz, mha, mhd", required=True)
    parser.add_argument("--dest", "-o", "--output", type=str, help="Output directory for results.", required=True)
    parser.add_argument("--model", type=str, default=None, help="Model key for prediction, defaults to 'ts2d-v1-ep4000b2'.")
    parser.add_argument("--no-remote", action="store_true", help="Disable remote model download. Models must be available locally.")
    parser.add_argument("--no-fetch", action="store_true", help="Disable to not fetch the latest model URLs from the remote repository and use the local shared.json instead.")
    parser.add_argument("--collapse", action="store_true", help="Collapse projected images to 2D. This removes the 3D geometrical information.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results as PNG images.")
    parser.add_argument("--save-all", action="store_true", help="In addition to the final result, also saves results for each individual model.")
    parser.add_argument("--silent", action="store_true", help="Hides any unnecessary output.")

    args = parser.parse_args()

    ts2d_run(
        src=args.src,
        dest=args.dest,
        model=args.model,
        use_remote=not args.no_remote,
        fetch_remote=not args.no_fetch,
        collapse=args.collapse,
        visualize=args.visualize,
        save_all=args.save_all,
        silent=args.silent
    )

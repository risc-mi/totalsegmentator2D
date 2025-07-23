import sys

import SimpleITK as sitk
import numpy as np

from ts2d.core.util.log import warn
from ts2d.core.util.sitk_util import get_affine_transform
from ts2d.core.util.types import native, default
from ts2d.core.util.util import parse_float, unit_vector


def axis_name_to_index(name: str) -> int:
    """
    Maps an axis name to its respective axis index in RAI coordinates
    """
    return {
        'a': 2,
        'ax': 2,
        'axial': 2,
        's': 0,
        'sag': 0,
        'sagittal': 0,
        'c': 1,
        'cor': 1,
        'coronal': 1,
    }[name.lower()]

def reorient_image(img: sitk.Image, orient: str = 'RAI') -> sitk.Image:
    """
    Reorients the directions to the specified orientation using sitk.DICOMOrient
    2D images are ignored
    :param img: image to reorient
    :param orient: orientation, defaults to RAI
    :return:
    """
    if img.GetDimension() > 2:
        from ts2d.core.util.meta import copy_image_meta
        return copy_image_meta(sitk.DICOMOrient(img, orient), img)
    return img


def project(img: sitk.Image, mode='max', axis: int | str = -1) -> sitk.Image:
    """
    Projects an image, by default along the last axis.
    possible projection modes are:
    - 'first': the first value that is not zero along the projection axis
    - 'max': maximum intensity (default)
    - 'min': minimum intensity
    - 'avg' or 'mean': mean intensity
    - 'median': median intensity
    - 'std': standard deviation of intensities
    - 'depth': depth projection, same as 'first'
    - 'multiclass': multiclass projection for labels, i.e., in separate channels
    - 'slice': extract a 2d slice instead of a projection, works with first, middle, last or a float value between 0.0 and 1.0
    :param img: image to project
    :param mode: projection mode, defaults to 'max'
    :param axis: axis to project along, defaults to the last one
    :return: projected image
    """
    axis = axis_name_to_index(axis) if isinstance(axis, str) else list(range(img.GetDimension()))[axis]
    mode = str(mode).lower().strip()
    mode, *param = f'{mode}:'.split(':')[:-1]
    op = {
        'first': lambda x: _project_first(x, axis=axis),
        'max': sitk.MaximumProjectionImageFilter,
        'mip': sitk.MaximumProjectionImageFilter,
        'min': sitk.MinimumProjectionImageFilter,
        'avg': sitk.MeanProjectionImageFilter,
        'mean': sitk.MeanProjectionImageFilter,
        'median': sitk.MedianProjectionImageFilter,
        'std': sitk.StandardDeviationProjectionImageFilter,
        'depth': lambda x: _project_first(x, axis=axis),
        'multiclass': lambda x: _project_multiclass(x, num=param[0], axis=axis),
        'slice': lambda x: _extract_slice(x, pos=param[0], axis=axis)
    }.get(mode, None)

    if op is None:
        raise RuntimeError("Unsupported filter mode: {}".format(mode))
    if isinstance(op, type) and issubclass(op, sitk.ImageFilter):
        filter = op()
        filter.SetProjectionDimension(axis)
        res = filter.Execute(img)
    else:
        res = op(img)

    # reset the origin
    origin = list(res.GetOrigin())
    origin[axis] = img.GetOrigin()[axis]
    res.SetOrigin(origin)
    return res

def extract_slice_index(img: sitk.Image, index: int, axis: int = -1):
    """
    extracts a slice from the specified image
    :param img: img to crop
    :param index: slice index to extract from the axis
    :param axis: axis to slice
    :return: the extracted slice
    """
    dim = img.GetDimension()
    if not (-dim <= axis < dim):
        raise RuntimeError("The specified axis {} is not valid for an image of dimensionality: {}".format(axis, dim))

    size = img.GetSize()
    n_slices = size[axis]
    if not (0 <= index < n_slices):
        raise RuntimeError("Slice index is outside the available range: [0, {}]".format(n_slices-1))

    roi_index = [0] * img.GetDimension()
    roi_index[axis] = index
    roi_size = list(size)
    roi_size[axis] = 1

    return sitk.Extract(img, roi_size, roi_index)

def extract_slice_factor(img: sitk.Image, pos: float, axis: int = -1):
    """
    extracts a slice from the specified image, uses a factor between 0.0 and 1.0 instead of an index position.
    :param img: img to crop
    :param index: slice position to extract a slice from, between 0.0 (origin) and 1.0 (end)
    :param axis: axis to slice
    :return: the extracted slice
    """
    size = img.GetSize()
    n_slices = size[axis]
    index = int(np.clip(np.round(n_slices * pos), a_min=0, a_max=n_slices))
    return extract_slice_index(img, index=index, axis=axis)


def _project_first(img: sitk.Image, axis=0):
    arr = sitk.GetArrayFromImage(img).T
    shape = list(arr.shape)
    shape[axis] = 1
    arr = np.moveaxis(arr, axis, 0) if axis != 0 else arr
    idx = np.argmax(arr != 0, axis=0).flatten()
    arr = arr.reshape(arr.shape[0], -1).T[np.arange(idx.size), idx]
    arr = arr.reshape(shape)
    res = sitk.GetImageFromArray(arr.T)
    res.CopyInformation(sitk.MinimumProjection(img, axis))
    return res

def _extract_slice(img: sitk.Image, pos='first', axis=0):
    factor = parse_float(pos, err=None)
    if factor is None:
        factor = {
            'first': 0,
            'middle': 0.5,
            'last': 1
        }.get(pos, None)
    assert factor is not None, "Invalid slice position: {}".format(pos)
    return extract_slice_factor(img, pos=factor, axis=axis)

def _project_multiclass(img: sitk.Image, num, axis=0):
    channels = img.GetNumberOfComponentsPerPixel()
    if channels == 1:
        num = int(num)
        arr = sitk.GetArrayFromImage(img).T
        dims = img.GetDimension()
        axis = tuple(range(dims))[axis]

        # move the target axis to the front
        shape = list(arr.shape)
        shape[axis] = 1
        arr = np.moveaxis(arr, axis, 0) if axis != 0 else arr

        idx = np.nonzero(arr)
        slices = (arr[idx] - 1, 0) + idx[1:]
        arr = np.zeros((num, 1) + arr.shape[1:], dtype=np.uint8)
        if np.any(idx):
            arr[slices] = 1

        # move the target axis to its original position
        arr = np.moveaxis(arr, 1, axis+1) if axis != 0 else arr

        # convert to an image and set the same information as sitk would for projection
        res = sitk.GetImageFromArray(arr.T, isVector=True)
        res.CopyInformation(sitk.BinaryProjection(img, axis))
    else:
        # this is already a multichannel image, lets just project it
        # BinaryProjection was not supported for vector images in SimpleITK 2.4.0
        from ts2d.core.util.meta import copy_image_meta
        res = copy_image_meta(sitk.MaximumProjection(img, axis), img)
    return res

def read_image_nibabel(fp: str) -> sitk.Image:
    """
    uses nibabel to read an image
    use for images that SimpleITK fails to load, e.g., because of a missing cosine direction matrix
    Note: originally implemented to handle the TotalSegmentator dataset
    :param fp: path to the image to load
    :return: image loaded with nibabel and converted to a SimpleITK Image
    """
    import nibabel as nib
    import numpy as np

    nib_img = nib.load(fp)
    arr = nib_img.get_fdata().T
    img = sitk.GetImageFromArray(arr)
    aff = nib_img.affine
    aff = np.matmul(np.diag([-1., -1., 1., 1.]), aff)

    tf = get_affine_transform(aff)
    spacing = native(np.asarray(nib_img.header.get_zooms(), dtype=float))
    origin = tf.TransformPoint([0, 0, 0])
    tf = tf.GetInverse()
    dirs = tuple(unit_vector(tf.TransformVector(v, [0, 0, 0])) for v in np.eye(3))

    # Set the extracted properties to the SimpleITK image
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(np.asarray(dirs).flatten().tolist())
    return img

def read_image(fp: str) -> sitk.Image:
    try:
        return sitk.ReadImage(fp)
    except RuntimeError as ex:
        if 'orthonormal' in str(ex):
            warn(f"Known ITK image loading issue regarding orthonormal direction cosines occurred: {ex}")
            try:
                import nibabel
            except ImportError:
                warn("Please install nibabel to enable the fallback image loading method (pip install nibabel)")
                sys.exit(1)
            warn("Using the fallback image reading method based on nibabel...")
            return read_image_nibabel(fp)
        raise


def reduce_dimensions(img: sitk.Image, strategy=None, min=None) -> sitk.Image:
    """
    Collapses dimensions of size 1
    :param img: image to reduce dimensions of
    :param min: a minimum of dimensions to keep
    """
    strategy = default(strategy, sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOGUESS)
    index = [0] * img.GetDimension()
    size = list((s if s > 1 else 0) for s in img.GetSize())
    if min:
        refill = min-sum(s > 0 for s in size)
        for idx in range(-1, -len(size)-1, -1):
            if refill <= 0:
                break
            if size[idx] == 0:
                size[idx] = 1
                refill -= 1
    return sitk.Extract(img, size, index, strategy)
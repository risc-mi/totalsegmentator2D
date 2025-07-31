import sys
from typing import List

import SimpleITK as sitk
import numpy as np

from ts2d.core.util.color import to_palette
from ts2d.core.util.log import warn
from ts2d.core.util.meta import copy_image_meta, copy_image_geo, get_annotation_labels, get_labels, set_annotation_meta, \
    get_label_mask
from ts2d.core.util.sitk_util import get_affine_transform, is_label_image, is_label_type
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

def convert_img_dims(ref: sitk.Image, v):
    """
    for a value that can be a tuple or scalar, return a spacing tuple of the reference image dimension
    """
    return [v] * ref.GetDimension() if np.isscalar(v) else v

def image_vector_flatten_max(img: sitk.Image, index=False) -> sitk.Image:
    """
    uses numpy to flatten a vector image (if it has more than one component), i.e., remove the component dimension
    each voxel gets the maximum value over all of its component values
    Note: supports multichannel labelmaps
    :param img: simple itk image to flatten, a vector image
    :param index: set the index of the maximum nonzero component instead
    :return: flat image with a single component
    """
    if img.GetNumberOfComponentsPerPixel() > 1:
        if index:
            arr = sitk.GetArrayFromImage(img).T
            mask = np.vstack([np.ones((1, )+arr.shape[1:], dtype=bool), arr != 0], dtype=bool)
            mask = mask[::-1, ...]
            idx = np.argmax(mask, axis=0)
            arr = arr.shape[0] - idx
            arr = arr.T
        else:
            arr = np.max(sitk.GetArrayFromImage(img), axis=-1)

        res = sitk.GetImageFromArray(arr)
        copy_image_geo(res, img)
        copy_image_meta(res, img)
        return res
    return img


def resample(img, spacing, labels=None, size=None, interpolation=None, center=None, center_position=None, default_value=0, extrapolate=False,
             setCallback=None) -> sitk.Image:
    """
    Resamples an image to the specified spacing (in mm)
    Sometimes, it is necessary to enforce an output image size: in this case, size and center can be specified to automatically
    crop the image in the resampling step - this is more effective and accurate than cropping in a postprocessing-step).
    :param img: image to resample
    :param spacing: spacing in mm to target
    :param labels: whether the image contains labels (mask or segmentation), this affects the default interpolation
    :param size: (optional) size of the output image to enforce (relative to the center), if not specified, the size is calculated
    :param interpolation: (optional) SimpleITK interpolation method, if not specified, the default is determined from the image
    :param center: center index of the image, if not specified, the center will not change
    :param center_position: center position of the image, alternative to the argument center (which is an index)
    :param default_value: fill value for sampled areas outside the original extent
    :param extrapolate: whether to enable extrapolation (outside the volume)
    :return: resampled image
    """
    filter = resample_filter(img=img, spacing=spacing, labels=labels, size=size, interpolation=interpolation, center=center, center_position=center_position,
                             default_value=default_value, extrapolate=extrapolate)
    if filter is None:
        return img
    if setCallback is not None:
        setCallback(filter)
    return filter.Execute(img)


def resample_filter(img, spacing, labels=None, size=None, interpolation=None, center=None, center_position=None, default_value=0, extrapolate=False):
    """
    Instantiates the SimpleITK filter based on the specified image
    For the arguments see resample
    """
    spacing = convert_img_dims(img, spacing)
    old_spacing = img.GetSpacing()
    new_spacing = [v for v in spacing]
    old_size = img.GetSize()
    if size is None or (None in size):
        auto_size = [int(0.5 + old_size[i] * s / new_spacing[i]) for i, s in enumerate(old_spacing)]
        size = auto_size if size is None else tuple((_a if _s is None else _s) for _s, _a in zip(size, auto_size))
    if center is None:
        if center_position is None:
            center = np.multiply(old_size, 0.5)
    else:
        if center_position is not None:
            raise RuntimeError("Either center or center_position may be specified - not both!")
    if center_position is None:
        center_position = img.TransformIndexToPhysicalPoint(np.asarray(center, dtype=int).tolist())

    ref = sitk.Image(native(size), img.GetPixelID())
    ref.SetDirection(img.GetDirection())
    ref.SetSpacing(native(new_spacing))

    new_diff = np.subtract(ref.TransformIndexToPhysicalPoint(np.multiply(size, 0.5).astype(int).tolist()), ref.GetOrigin())
    new_origin = center_position-new_diff
    ref.SetOrigin(new_origin)

    trans = sitk.Transform()
    trans.SetIdentity()
    if labels is None:
        labels = is_label_type(img.GetPixelIDValue())
    if interpolation is None:
        interpolation = sitk.sitkBSpline if not labels else sitk.sitkNearestNeighbor

    if img.GetPixelIDValue() == sitk.sitkUInt8 and interpolation != sitk.sitkNearestNeighbor:
        warn("SimpleITK does not permit interpolation of UInt8 images, falling back to nearest neighbour.")
        interpolation = sitk.sitkNearestNeighbor

    changed = not np.allclose(spacing, old_spacing)
    if not changed:
        changed = ref.GetSize() != img.GetSize() or ref.GetOrigin() != img.GetOrigin()
    if changed:
        filter = sitk.ResampleImageFilter()
        filter.SetReferenceImage(ref)
        filter.SetTransform(trans)
        filter.SetOutputPixelType(img.GetPixelID() if not labels else sitk.sitkUInt8)
        filter.SetInterpolator(interpolation)
        filter.SetDefaultPixelValue(default_value)
        filter.SetUseNearestNeighborExtrapolator(extrapolate)
        return filter
    else:
        return None

def resample_uniform(img, **kwargs) -> sitk.Image:
    """
    Returns the image in uniform spacing, resampling the larger resolutions to smallest one
    :param kwargs: for further arguments see resample
    """
    spacing = min(img.GetSpacing())
    return resample(img, spacing, **kwargs)


def create_visual(img: sitk.Image, mode='max', axis: int|str=-1,
              window=None, labels=None, palette=None) -> sitk.Image:
    """
    creates a 2d visualization .png of img, if necessary using projections to reduce the dimensionality
    Window methods:
    - minmax: adjust the window to the minimum and maximum values
    - pc5: use the 5th and 95th percentile
    :param img: nd image
    :param mode: if necessary, mode to use for projection, defaults to 'max'
    :param axis: axis to project to, defaults to the last axis (-1)
    :param window: (optional) intensity window to use for both images, as a tuple of (min, max)
    :param labels: whether the image contains labels, i.e., is a segmentation, which influences resampling, defaults to None, in which case the method tries to determine the type
    :param palette: for label images, colors to match label values too
    """
    try:
        labels = default(labels, palette or is_label_image(img))
    except:
        labels = False
    if labels and not palette:
        # try to extract the palette from the meta information
        try:
            palette = dict()
            meta = get_annotation_labels(img)
            for k, v in meta.items():
                value, color = v.get('value'), v.get('color')
                if value is not None and color is not None:
                    palette[int(value)] = color
        except Exception as ex:
            warn(f"Failed to extract palette from image metadata: {ex}")
            pass

    img = reorient_image(img)

    _axis = axis_name_to_index(axis) if isinstance(axis, str) else default(axis, -1)
    while True:
        img = reduce_dimensions(img, min=2)
        dim = img.GetDimension()
        if dim <= 2:
            break
        _axis = -1 if abs(_axis) > dim else _axis
        img = project(img, mode=mode, axis=_axis)

    if labels:
        map = []
        if palette is not None:
            map = to_palette(palette)
        if img.GetNumberOfComponentsPerPixel() > 1:
            # convert the label map to label values
            img = image_vector_flatten_max(img, index=True)

        img = resample_uniform(img, labels=labels)
        img = sitk.LabelToRGB(img, 0, np.asarray(map).flatten().tolist())
    else:
        img = resample_uniform(img, labels=labels)
        window = get_auto_window(img, window) if (window is None or isinstance(window, str)) else window
        lower, upper = window
        if lower is None or upper is None:
            try:
                stats = sitk.MinimumMaximumImageFilter()
                stats.Execute(img)
                lower = stats.GetMinimum() if lower is None else lower
                upper = stats.GetMaximum() if upper is None else upper
            except:
                lower = 0
                upper = 255
        if img.GetNumberOfComponentsPerPixel() > 1:
            img = sitk.VectorMagnitude(img)
        img = sitk.IntensityWindowing(img, lower, upper, 0, 255)
        img = sitk.Cast(img, sitk.sitkUInt8)

    return img

def get_auto_window(img, method):
    if method is None:
        method = 'minmax'

    method = method.lower()
    if method == 'minmax':
        stat = sitk.MinimumMaximumImageFilter()
        stat.Execute(img)
        res = stat.GetMinimum(), stat.GetMaximum()
    elif method.startswith('pc'):
        pcstr = method.removeprefix('pc')
        try:
            if '-' in pcstr:
                pc = tuple(float(a) for a in pcstr.split('-'))
            else:
                pc = float(pcstr)
                pc = (pc, 100-pc)
        except:
            raise RuntimeError("Failed to parse percentile value from windowing method: {}".format(method))
        if len(pc) > 2:
            raise RuntimeError("The percentile can only be a range value: found value {}".format(method))
        aimg = sitk.GetArrayViewFromImage(img)
        res = native(np.percentile(aimg, pc))
    else:
        raise RuntimeError("Unkown windowing method: {}".format(method))

    return res

def get_actual_dimension(img: sitk.Image):
    """
    Returns actual dimensionality of an image, discarding dimensions of size 1
    """
    return sum(s > 1 for s in img.GetSize())


def combine_segmentations(segs: List[sitk.Image]):
    """
    combines multiple segmentations into a single one
    """
    res = list()
    names = dict()
    colors = dict()
    for seg in segs:
        seg_labels = get_annotation_labels(seg)
        for name, info in seg_labels.items():
            mask = get_label_mask(seg, info['value'])
            idx = len(res)
            names[idx+1] = name
            c = info.get('color')
            if c is not None:
                colors[name] = c
            res.append(mask)

    res = sitk.Compose(res)
    set_annotation_meta(res, names=names, colors=colors)
    return res

def split_channels(img: sitk.Image):
    """
    Splits the component channels of an image into a list of single-channel images.
    :param img: SimpleITK image
    :return: list of single-channel images
    """
    nch = img.GetNumberOfComponentsPerPixel()
    chs = [img] if nch == 1 else [sitk.VectorIndexSelectionCast(img, i) for i in range(nch)]
    return chs
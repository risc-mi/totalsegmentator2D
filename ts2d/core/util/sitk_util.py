import SimpleITK as sitk
import numpy as np

from ts2d.core.util.types import native

_sitk_types_int_signed = {sitk.sitkInt8, sitk.sitkInt16, sitk.sitkInt32, sitk.sitkInt64}
_sitk_types_int_unsigned = {sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkUInt32, sitk.sitkUInt64}
_sitk_types_labels = {sitk.sitkLabelUInt8, sitk.sitkLabelUInt16, sitk.sitkLabelUInt32, sitk.sitkLabelUInt64}
_sitk_types_vint_signed = {sitk.sitkVectorInt8, sitk.sitkVectorInt16, sitk.sitkVectorInt32, sitk.sitkVectorInt64}
_sitk_types_vint_unsigned = {sitk.sitkVectorUInt8, sitk.sitkVectorUInt16, sitk.sitkVectorUInt32, sitk.sitkVectorUInt64}

_sitk_types_float = {sitk.sitkFloat32, sitk.sitkFloat64}
_sitk_types_complex = {sitk.sitkComplexFloat32, sitk.sitkComplexFloat64}
_sitk_types_vfloat = {sitk.sitkVectorFloat32, sitk.sitkVectorFloat64}


def is_label_type(t):
    """
    returns true if t is a sitk label pixel type (including regular UInt8)
    :param t: sitk pixel type
    """
    global _sitk_types_labels
    return t in _sitk_types_labels or t == sitk.sitkUInt8


def is_label_image(img: sitk.Image):
    """
    returns true if img is of a label pixel type (including regular UInt8)
    :param img: image to check
    """
    return is_label_type(img.GetPixelIDValue())

def get_affine_transform(aff: np.ndarray):
    """
    returns a simpleitk affine transform for a homological affine matrix (affine transform + translation)
    :param aff: numpy homological matrix
    :return: simpleitk affine transform
    """
    aff = np.asarray(aff)
    dim = np.unique(aff.shape)
    if dim.size != 1:
        raise RuntimeError("The affine matrix must be a square matrix: got: got {}"
                           "".format('x'.join(str(s) for s in aff.shape)))
    dim = native(dim[0])-1
    res = sitk.AffineTransform(dim)
    res.SetMatrix(native(aff[:dim, :dim].flatten()))
    res.SetTranslation(native(aff[:dim, dim]))
    return res
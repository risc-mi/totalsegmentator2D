import copy
from typing import Dict, Union, Optional

import SimpleITK as sitk
import numpy as np

from ts2d.core.util.log import warn
from ts2d.core.util.sitk_util import is_label_image
from ts2d.core.util.util import short_message
from ts2d.core.util.types import default, as_set


def get_labels(seg: sitk.Image, bg=None, numpy=False, fetch=True) -> list:
    """
    returns a sorted list of labels in the segmentation mask/image, not counting any background label
    Note: supports multichannel labelmaps
    :param seg label/mask image to process
    :param bg one or many labels to disregard as background, defaults to the label value 0
    :param numpy whether to use numpy or simpleitk to determine the labels, numpy can be considerably slower
    :param fetch: check whether the label is actually set/present in the segmentation, only appicable for multicomponent segmentations
    """
    bg = default(bg, 0)
    bg = as_set(bg)
    if seg.GetNumberOfComponentsPerPixel() > 1:
        # multi-component segmentation, return the number of components
        if fetch:
            res = list(i+1
                       for i, b in enumerate(np.any(sitk.GetArrayViewFromImage(seg),
                                                    axis=tuple(range(seg.GetDimension()))))
                       if b)
        else:
            res = list(i+1 for i in range(seg.GetNumberOfComponentsPerPixel()))
    else:
        if not numpy:
            if not is_label_image(seg):
                seg = sitk.Cast(seg, sitk.sitkInt64)

            f = sitk.LabelShapeStatisticsImageFilter()
            f.ComputeFeretDiameterOff()
            f.ComputePerimeterOff()
            f.ComputeOrientedBoundingBoxOff()
            f.Execute(seg)
            res = f.GetLabels()
        else:
            res = tuple(int(l) for l in np.unique(sitk.GetArrayViewFromImage(seg)))
    res = set(res).difference(bg)
    return sorted(res)

def copy_image_meta(dst: sitk.Image, src: [sitk.ImageFileReader, sitk.Image]):
    """
    copies metadata from the source to the destination image
    :param dst: destination image
    :param src: source image
    """
    set_image_meta(dst, get_image_meta(src))
    return dst

def _sanitize_meta(meta: dict):
    meta.pop("6000|3000", None)
    return meta

def get_image_meta(img: Union[sitk.Image, sitk.ImageReaderBase, str],
                   add_info=False,
                   orient=None,
                   slices=False,
                   sanitize=False) -> dict:
    """
    returns the images meta data as a dictionary
    :param add_info: whether to add image information to the metadata
    :param orient: add addition information about the image orientation
    :param slices: if True, meta information will be loaded for each slice (only for image readers)
    :param sanitize: clears tags with large irrelevant data
    :return: meta data as a dictionary
    """
    if isinstance(img, str):
        reader = sitk.ImageFileReader()
        reader.SetFileName(img)
        reader.ReadImageInformation()
        if orient is None:
            img = reader
        else:
            # only necessary when advanced info is extracted and the DICOM orientation matters
            img = reader.Execute()
            from ts2d.core.util.image import reorient_image
            img = reorient_image(img, orient)

    if isinstance(img, sitk.ImageSeriesReader):
        slice_indices = range(len(img.GetFileNames()) if slices else 1)
        slice_meta = dict((f'slice{idx}', dict((key, img.GetMetaData(idx, key))
                                         for key in img.GetMetaDataKeys(idx)))
                    for idx in slice_indices)
        meta = copy.deepcopy(slice_meta['slice0'])
        if slices:
            if sanitize:
                slice_meta = dict((k, _sanitize_meta(m)) for k, m in slice_meta.items())
            meta['slices'] = slice_meta
    else:
        meta = dict((key, img.GetMetaData(key))
                    for key in img.GetMetaDataKeys())
    if add_info:
        add_info_meta(img, meta)

    meta = _sanitize_meta(meta) if sanitize else meta

    return meta

def set_image_meta(img: sitk.Image, meta: dict, limit: Optional[int] = None, clear=False):
    """
    writes values from a dictionary to image metadata
    limit: character limit for attributes
    clear: whether to clear the existing metadata
    """
    if clear:
        for key in img.GetMetaDataKeys():
            img.EraseMetaData(key)

    for key, value in meta.items():
        try:
            key = key.encode('utf-8', 'namereplace').decode('utf-8')
            value = value.encode('utf-8', 'namereplace').decode('utf-8')
            if limit:
                value = short_message(value, limit)
            img.SetMetaData(key, value)
        except UnicodeEncodeError as ex:
            pass


def set_annotation_meta(seg: sitk.Image, names: Dict[int, str]=None, colors: Union[str, Dict[str, str]]=None, combined: Dict[int, dict]=None):
    """
    sets the relevant meta information to identify segments in the segmentation image
    supports regular (single channel) segmentations and multilabel (multi-channel/one-hot) segmentations
    Note: supports multichannel labelmaps
    :param seg: segmentation to write metadata to
    :param names: lookup from label value to name
    :param colors: lookup from label name to color
    """
    info = get_image_meta(seg)

    multilabel = seg.GetNumberOfComponentsPerPixel() > 1
    labels = list(range(seg.GetNumberOfComponentsPerPixel())) if multilabel else get_labels(seg)

    if names is not None and (colors is None or isinstance(colors, str)):
        try:
            colors_name = colors
            from ts2d.core.util.color import named_palette
            colors = dict(enumerate(named_palette(colors, max(labels)+1)))
            colors = dict((names[key], colors[key]) for key in names.keys() if key in labels)
        except:
            warn("Failed to load named palette: {}".format(colors_name))

    # remove any existing segment information
    for k in list(info.keys()):
        if k.startswith('Segment'):
            info.pop(k)

    # write the new segment information
    for seg_id, label in enumerate(labels):
            pattern = 'Segment{}_{}'.format(seg_id, '{}')
            if names is not None:
                info.update({
                    pattern.format('ID'): str(seg_id),
                    pattern.format('Layer'): str(label if multilabel else 0),
                    pattern.format('LabelValue'): str(1 if multilabel else label),
                })

                name = names.get(label + 1 if multilabel else label)
                if name is not None:
                    info.update({
                        pattern.format('Name'): str(name),
                        pattern.format('NameAutoGenerated'): '0'
                    })

                    try:
                        color = colors.get(name)
                        if color is not None:
                            from ts2d.core.util.color import to_color_str_rgb_floats
                            color = to_color_str_rgb_floats(color, sep=' ')
                            info.update({
                                pattern.format('Color'): str(color),
                                pattern.format('ColorAutoGenerated'): '0'
                            })
                    except:
                        warn("Failed to read a color value for label {} (value {})".format(name, label))
            elif combined is not None:
                linfo = combined.get(label)
                if linfo:
                    for k, v in linfo.items():
                        info[pattern.format(k)] = str(v if not isinstance(v, bool) else int(v))

    # set the metadata and make sure existing metadata is cleared
    set_image_meta(seg, info, clear=True)

def add_info_meta(img: Union[sitk.Image, sitk.ImageReaderBase, str], meta: dict, text=False):
    info = dict()
    info['size'] = img.GetSize()
    info['spacing'] = img.GetSpacing()
    info['direction'] = img.GetDirection()
    info['dimension'] = img.GetDimension()
    info['origin'] = img.GetOrigin()
    info['pixelid'] = img.GetPixelIDValue()
    if text:
        info = dict((k, str(v)) for k, v in info.items())
    meta.update(info)
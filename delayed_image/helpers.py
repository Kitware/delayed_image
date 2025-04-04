import kwimage
import ubelt as ub
import numpy as np
import math
from delayed_image.util import util_network_text


write_network_text = util_network_text.write_network_text


def _auto_dsize(transform, sub_dsize):
    """
    Returns:
        Tuple[int, int]

    Example:
        transform = kwimage.Affine.random()
        sub_dsize = (512, 512)
    """
    sub_w, sub_h = sub_dsize

    if 0:
        sub_bounds = kwimage.Coords(
            np.array([[0,     0], [sub_w, 0],
                      [0, sub_h], [sub_w, sub_h]])
        )
        bounds = sub_bounds.warp(transform.matrix)
        max_xy = np.ceil(bounds.data.max(axis=0))
    else:
        # note: this is faster than the above variant but will break on
        # non-affine (i.e. homogenous) transforms.
        sub_bounds = np.array([
            [0,     0, 1],
            [sub_w, 0, 1],
            [0, sub_h, 1],
            [sub_w, sub_h, 1]
        ])
        # bounds = kwimage.warp_points(transform.matrix, sub_bounds)[0:2]
        bounds = (transform.matrix[0:2] @ sub_bounds.T).T
        max_xy = np.ceil(bounds.max(axis=0))
    max_x = int(max_xy[0])
    max_y = int(max_xy[1])
    dsize = (max_x, max_y)
    return dsize


def _ensure_valid_dsize(dsize):
    """
    Ensure dsize members are ints or Nones

    Returns:
        None | Tuple[int | None, int | None]
    """
    if dsize is not None:
        if ub.iterable(dsize):
            w, h = dsize
            if w is not None:
                w = int(w)
            if h is not None:
                h = int(h)
            dsize = (w, h)
    return dsize


def _ensure_valid_shape(shape):
    """
    Ensure dsize members are ints or Nones

    Returns:
        Tuple[int, ...]
    """
    return tuple(int(s) for s in shape)


def _largest_shape(shapes):
    """
    Finds maximum over all shapes

    Example:
        >>> shapes = [
        >>>     (10, 20), None, (None, 30), (40, 50, 60, None), (100,)
        >>> ]
        >>> largest = _largest_shape(shapes)
        >>> print('largest = {!r}'.format(largest))
        >>> assert largest == (100, 50, 60, None)
    """
    def _nonemax(a, b):
        if a is None or b is None:
            return a or b
        return max(a, b)
    import itertools as it
    largest = []
    for shape in shapes:
        if shape is not None:
            largest = [
                _nonemax(c1, c2)
                for c1, c2 in it.zip_longest(largest, shape, fillvalue=None)
            ]
    largest = tuple(largest)
    return largest


def _swap_warp_after_crop(in_crop2, in_warp1, origin_convention='corner'):
    r"""
    Given a warp followed by a crop, compute the corresponding crop followed by
    a warp.

    Given a region in a "root" image and a trasnform between that "root" and
    some "leaf" image, compute the appropriate quantized region in the "leaf"
    image and the adjusted transformation between that root and leaf.

    Args:
        in_crop2 (kwimage.Polygon):
            region representing the crop that happens after the warp

        in_warp1 (kwimage.Affine):
            the warp that happens before the input crop

    Returns:
        Tuple[Tuple[slice, slice], kwimage.Affine]:
            out_crop1 - the crop that happens before the warp
            out_warp2 - warp that happens after the crop.

    Example:
        >>> from delayed_image.helpers import *  # NOQA
        >>> from delayed_image.helpers import _swap_warp_after_crop
        >>> region_slices = (slice(33, 100), slice(22, 62))
        >>> region_shape = (100, 100, 1)
        >>> root_region_box = kwimage.Boxes.from_slice(region_slices, shape=region_shape)
        >>> root_region_bounds = root_region_box.to_polygons()[0]
        >>> tf_leaf_to_root = kwimage.Affine.affine(scale=7).matrix
        >>> slices, tf_new = _swap_warp_after_crop(root_region_bounds, tf_leaf_to_root)
        >>> print('tf_new =\n{!r}'.format(tf_new))
        >>> print('slices = {!r}'.format(slices))

    Example:
        >>> from delayed_image.helpers import *  # NOQA
        >>> from delayed_image.helpers import _swap_warp_after_crop
        >>> region_slices = (slice(0, 8), slice(0, 8))
        >>> region_shape = (100, 100, 1)
        >>> root_region_box = kwimage.Boxes.from_slice(region_slices, shape=region_shape)
        >>> root_region_bounds = root_region_box.to_polygons()[0]
        >>> tf_leaf_to_root = kwimage.Affine.affine(scale=2.0001).matrix
        >>> slices, tf_new = _swap_warp_after_crop(root_region_bounds, tf_leaf_to_root)
        >>> print('tf_new =\n{!r}'.format(tf_new))
        >>> print('slices = {!r}'.format(slices))

    Example:
        >>> from delayed_image.helpers import *  # NOQA
        >>> from delayed_image.helpers import _swap_warp_after_crop
        >>> region_slices = (slice(0, 8), slice(0, 8))
        >>> region_shape = (100, 100, 1)
        >>> root_region_box = kwimage.Boxes.from_slice(region_slices, shape=region_shape)
        >>> in_warp1 = kwimage.Affine.affine(scale=0.4).matrix
        >>> in_crop2 = root_region_box.to_polygons()[0]
        >>> out_crop1, out_warp2 = _swap_warp_after_crop(in_crop2, in_warp1)
        >>> print('in_warp1 =\n{!r}'.format(in_warp1))
        >>> print('in_crop2 = {!r}'.format(in_crop2.box().to_slice()))
        >>> print('out_crop1 = {!r}'.format(out_crop1))
        >>> print('out_warp2 =\n{!r}'.format(out_warp2))

    Example:
        >>> from delayed_image.helpers import *  # NOQA
        >>> from delayed_image.helpers import _swap_warp_after_crop
        >>> region_slices = (slice(0, 8), slice(4, 12))
        >>> region_shape = (100, 100, 1)
        >>> root_region_box = kwimage.Boxes.from_slice(region_slices, shape=region_shape)
        >>> in_warp1 = kwimage.Affine.affine(scale=4).matrix
        >>> in_crop2 = root_region_box.to_polygons()[0]
        >>> out_crop1, out_warp2 = _swap_warp_after_crop(in_crop2, in_warp1, origin_convention='center')
        >>> center_items = {
        >>>     'input': {'in_warp1': in_warp1, 'in_crop2': in_crop2.box().to_slice()},
        >>>     'output': {'out_crop1': out_crop1, 'out_warp2': out_warp2},
        >>> }
        >>> out_crop1, out_warp2 = _swap_warp_after_crop(in_crop2, in_warp1, origin_convention='corner')
        >>> corner_items = {
        >>>     'input': {'in_warp1': in_warp1, 'in_crop2': in_crop2.box().to_slice()},
        >>>     'output': {'out_crop1': out_crop1, 'out_warp2': out_warp2},
        >>> }
        >>> # xdoctest: +REQUIRES(module:rich)
        >>> import rich
        >>> rich.print(f'center_items = {ub.urepr(center_items, nl=2)}')
        >>> rich.print(f'corner_items = {ub.urepr(corner_items, nl=2)}')
    """
    # TODO: rename variables to make concepts more clear
    # Conceptually we want to warp the image and then crop out a box/polygon
    # inside the warped image. We will call these in_warp1, and in_crop1.  We
    # will then will produce a crop followed by a warp which is equivalent, but
    # more efficient. Will create out_crop1, out_warp2
    tf_leaf_to_root = in_warp1
    root_region_bounds = in_crop2

    needs_tl_corner_pad_workaround = origin_convention == 'corner'

    # Transform the region bounds into the sub-image space
    tf_leaf_to_root = kwimage.Affine.coerce(tf_leaf_to_root)
    tf_root_to_leaf = tf_leaf_to_root.inv()
    tf_root_to_leaf = tf_root_to_leaf.__array__()
    leaf_region_bounds = root_region_bounds.warp(tf_root_to_leaf)
    leaf_region_box = leaf_region_bounds.box().to_ltrb(copy=False)

    # Quantize to a region that is possible to sample from
    leaf_crop_box = leaf_region_box.quantize(inplace=True)

    # is this ok?
    # leaf_crop_box = leaf_crop_box.clip(0, 0, None, None, inplace=True)
    # Optimized clip
    ltrb = leaf_crop_box.data
    np.clip(ltrb, 0, None, out=ltrb)

    # Because we sampled a large quantized region, we need to modify the
    # transform to nudge it a bit to the left, undoing the quantization,
    # which has a bit of extra padding on the left, before applying the
    # final transform.
    # subpixel_offset = leaf_region_box.data[0, 0:2]
    crop_offset = leaf_crop_box.data[0:2]
    root_offset = root_region_bounds.exterior.data.min(axis=0)

    # TODO: could optimize this logic
    tf_root_to_newroot = kwimage.Affine.translate(offset=-root_offset).matrix
    tf_newleaf_to_leaf = kwimage.Affine.translate(offset=crop_offset).matrix

    # Resample the smaller region to align it with the root region
    # Note: The right most transform is applied first
    tf_newleaf_to_newroot = (
        tf_root_to_newroot @
        tf_leaf_to_root @
        tf_newleaf_to_leaf
    )

    lt_x, lt_y, rb_x, rb_y = leaf_crop_box.data

    if 1:
        # Candidate fix
        root_region_box = root_region_bounds.box()
        old_w = root_region_box.width
        old_h = root_region_box.height
        leaf_w = leaf_region_box.width
        leaf_h = leaf_region_box.height

        # TODO: test the case where old_w or old_h are zero
        # Attempt to work around issue #4
        # Not sure what the consequences of handing the issue this way are but
        # it seems to work.
        ZERO_WORKAROUND = 1

        if ZERO_WORKAROUND and old_w == 0:
            padw = 0
        else:
            padw = int(np.ceil(leaf_w / old_w))

        if ZERO_WORKAROUND and old_h == 0:
            padh = 0
        else:
            padh = int(np.ceil(leaf_h / old_h))

    else:
        padw, padh = 1, 1
    # padw, padh = 0, 0

    # # leaf_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))
    # print('root_region_bounds = {}'.format(ub.urepr(root_region_bounds, nl=1)))
    # print('leaf_region_bounds = {}'.format(ub.urepr(leaf_region_bounds, nl=1)))
    # print('tf_leaf_to_root = {}'.format(ub.urepr(tf_leaf_to_root, nl=1)))

    # Need to pad add a bit more to the end of the crop because we are going to
    # warp afterwards. This fixes at least one case of off-by-one error.
    rb_y_stop_index = rb_y + padh
    rb_x_stop_index = rb_x + padw

    if needs_tl_corner_pad_workaround:
        # pad by 1 if we are not on a left/top edge
        lt_pad_h = min(lt_y, 1)
        lt_pad_w = min(lt_x, 1)
        lt_y_start_index = lt_y - lt_pad_h
        lt_x_start_index = lt_x - lt_pad_w
        lt_offset = kwimage.Affine.translate((-lt_pad_w, -lt_pad_h))
        out_warp2 =  tf_newleaf_to_newroot @ lt_offset.matrix
    else:
        lt_y_start_index = lt_y
        lt_x_start_index = lt_x
        out_warp2 = tf_newleaf_to_newroot

    out_crop1 = (slice(lt_y_start_index, rb_y_stop_index),
                 slice(lt_x_start_index, rb_x_stop_index))
    return out_crop1, out_warp2


def _swap_crop_after_warp(inner_region, outer_transform):
    r"""
    Given a crop followed by a warp (usually an overview), compute the
    corresponding warp followed by a crop followed by a small correction warp.

    Note that in general it is not possible to ensure the crop is the last
    operation, there may need to be a small warp after it.

    However, this is generally only useful when the warp being pushed early in
    the operation chain corresponds to an overview, and often - but not always
    - the final warp will simply be the identity.

    Args:
        inner_region (kwimage.Polygon):
            region representing the crop that happens before the warp

        outer_transform (kwimage.Affine):
            the warp that happens after the input crop

    Returns:
        Tuple[kwimage.Affine, Tuple[slice, slice], kwimage.Affine]:

            new_inner_warp - the new warp to happen before the crop

            outer_crop - the new crop after the main warp

            new_outer_warp - a small subpixel alignment warp to happen last

    Example:
        >>> from delayed_image.helpers import *  # NOQA
        >>> region_slices = (slice(33, 100), slice(22, 62))
        >>> region_shape = (100, 100, 1)
        >>> inner_region = kwimage.Boxes.from_slice(region_slices)
        >>> inner_region = inner_region.to_polygons()[0]
        >>> outer_transform = kwimage.Affine.affine(scale=1/4)
        >>> new_inner_warp, outer_crop, new_outer_warp = _swap_crop_after_warp(inner_region, outer_transform)
        >>> print('new_inner_warp = {}'.format(ub.urepr(new_inner_warp, nl=1)))
        >>> print('outer_crop = {}'.format(ub.urepr(outer_crop, nl=1)))
        >>> print('new_outer_warp = {}'.format(ub.urepr(new_outer_warp, nl=1)))
    """
    # Find where the inner region maps to after the transform is applied
    outer_region = inner_region.warp(outer_transform)

    # Transform the region bounds into the sub-image space
    outer_box = outer_region.box().to_ltrb(copy=False)

    # Quantize to a region that is possible to sample from
    outer_crop_box = outer_box.quantize()

    # is this ok?
    outer_crop_box = outer_crop_box.clip(0, 0, None, None)

    # Because the new crop might not be perfectly aligned, we might need to
    # nudge it a bit after we crop out its bounds.
    crop_offset = outer_crop_box.data[0:2]
    outer_offset = outer_region.exterior.data.min(axis=0)

    # Compute the extra transform that will realign the quantized croped data
    # with the original warped inner crop.
    tf_crop_to_box = kwimage.Affine.affine(
        offset=crop_offset - outer_offset
    )

    lt_x, lt_y, rb_x, rb_y = outer_crop_box.data
    outer_crop = (slice(lt_y, rb_y), slice(lt_x, rb_x))
    new_outer_warp = tf_crop_to_box

    # The inner warp will be the same as the original outer warp.
    new_inner_warp = outer_transform

    return new_inner_warp, outer_crop, new_outer_warp


def dequantize(quant_data, quantization):
    """
    Helper for dequantization

    Args:
        quant_data (ndarray):
            data to dequantize

        quantization (Dict[str, Any]):
            quantization information dictionary to undo.
            Expected keys are:
            orig_dtype (str)
            orig_min (float)
            orig_max (float)
            quant_min (float)
            quant_max (float)
            nodata (None | int)

    Returns:
        ndarray : dequantized data

    Example:
        >>> quant_data = (np.random.rand(4, 4) * 256).astype(np.uint8)
        >>> quantization = {
        >>>     'orig_dtype': 'float32',
        >>>     'orig_min': 0,
        >>>     'orig_max': 1,
        >>>     'quant_min': 0,
        >>>     'quant_max': 255,
        >>>     'nodata': None,
        >>> }
        >>> dequantize(quant_data, quantization)

    Example:
        >>> quant_data = np.ones((4, 4), dtype=np.uint8)
        >>> quantization = {
        >>>     'orig_dtype': 'float32',
        >>>     'orig_min': 0,
        >>>     'orig_max': 1,
        >>>     'quant_min': 1,
        >>>     'quant_max': 1,
        >>>     'nodata': None,
        >>> }
        >>> dequantize(quant_data, quantization)
    """
    orig_dtype = quantization.get('orig_dtype', 'float32')
    orig_min = quantization.get('orig_min', 0)
    orig_max = quantization.get('orig_max', 1)
    quant_min = quantization.get('quant_min', 0)
    quant_max = quantization['quant_max']
    nodata = quantization.get('nodata', None)
    orig_extent = orig_max - orig_min
    quant_extent = quant_max - quant_min
    if quant_extent == 0:
        scale = 0
    else:
        scale = (orig_extent / quant_extent)
    dequant = quant_data.astype(orig_dtype)
    dequant = (dequant - quant_min) * scale + orig_min
    if nodata is not None:
        mask = quant_data == nodata
        dequant[mask] = np.nan
    return dequant


def quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.int16):
    """
    Note:
        Setting old_min / old_max indicates the possible extend of the input
        data (and it will be clipped to it). It does not mean that the input
        data has to have those min and max values, but it should be between
        them.

    Example:
        >>> from delayed_image.helpers import *  # NOQA
        >>> # Test error when input is not nicely between 0 and 1
        >>> imdata = (np.random.randn(32, 32, 3) - 1.) * 2.5
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))
        >>> #
        >>> for i in range(1, 20):
        >>>     print('i = {!r}'.format(i))
        >>>     quant2, quantization2 = quantize_float01(imdata, old_min=-i, old_max=i)
        >>>     recon2 = dequantize(quant2, quantization2)
        >>>     error2 = np.abs((recon2 - imdata)).sum()
        >>>     print('error2 = {!r}'.format(error2))

    Example:
        >>> # Test dequantize with uint8
        >>> from delayed_image.helpers import dequantize
        >>> imdata = np.random.randn(32, 32, 3)
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.uint8)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))

    Example:
        >>> # Test quantization with different signed / unsigned combos
        >>> print(quantize_float01(None, 0, 1, np.int16))
        >>> print(quantize_float01(None, 0, 1, np.int8))
        >>> print(quantize_float01(None, 0, 1, np.uint8))
        >>> print(quantize_float01(None, 0, 1, np.uint16))

    """
    # old_min = 0
    # old_max = 1
    quantize_iinfo = np.iinfo(quantize_dtype)
    quantize_max = quantize_iinfo.max
    if quantize_iinfo.kind == 'u':
        # Unsigned quantize
        quantize_nan = 0
        quantize_min = 1
    elif quantize_iinfo.kind == 'i':
        # Signed quantize
        quantize_min = 0
        quantize_nan = max(-9999, quantize_iinfo.min)

    quantization = {
        'orig_min': old_min,
        'orig_max': old_max,
        'quant_min': quantize_min,
        'quant_max': quantize_max,
        'nodata': quantize_nan,
    }

    old_extent = (old_max - old_min)
    new_extent = (quantize_max - quantize_min)
    quant_factor = new_extent / old_extent

    if imdata is not None:
        invalid_mask = np.isnan(imdata)
        new_imdata = (imdata.clip(old_min, old_max) - old_min) * quant_factor + quantize_min
        new_imdata = new_imdata.astype(quantize_dtype)
        new_imdata[invalid_mask] = quantize_nan
    else:
        new_imdata = None

    return new_imdata, quantization


class mkslice_cls:
    """
    Helper to make slice syntax easier to construct

    Example:
        >>> from delayed_image.helpers import mkslice_cls
        >>> m = mkslice_cls()
        >>> m[0:3]
        slice(0, 3, None)
        >>> m[0:3, 0:5]
        (slice(0, 3, None), slice(0, 5, None))
        >>> m()[0:3, 0:5]
        (slice(0, 3, None), slice(0, 5, None))
    """
    def __class_getitem__(self, index):
        # Doesnt exist in older Python versions
        return index
    def __getitem__(self, index):
        return index
    def __call__(self):
        return self

mkslice = mkslice_cls()


def _decompose_scale(self):
    """
    Scale only decomposition. Experimental method that is faster than
    decompose when only scale is needed.

    Args:
        self (kwimage.Affine): affine matrix to decompose
    """
    if self.matrix is None:
        return (1., 1.)
    a11, a12, _, a21, a22 = self.matrix.ravel()[0:5]
    # (a11, a12), (a21, a22) = self.matrix[0:2, 0:2]
    sx = math.sqrt(a11 * a11 + a21 * a21)
    theta = math.atan2(a21, a11)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    msy = a12 * cos_t + a22 * sin_t
    if abs(cos_t) < abs(sin_t):
        sy = (msy * cos_t - a12) / sin_t
    else:
        sy = (a22 - msy * sin_t) / cos_t
    scale = (sx, sy)
    return scale

"""
Intermediate operations
"""
import kwarray
import kwimage
import copy
import numpy as np
import ubelt as ub
import warnings
from delayed_image import delayed_base
from delayed_image import delayed_leafs
from delayed_image.channel_spec import FusedChannelSpec

# --------
# Stacking
# --------

__docstubs__ = """
from delayed_image.delayed_leafs import DelayedIdentity
from delayed_image.delayed_base import DelayedOperation
"""

TRACE_OPTIMIZE = 0  # TODO: make this a local setting
IS_DEVELOPING = 0  # set to 1 if hacking in IPython, otherwise 0 for efficiency


class DelayedArray(delayed_base.DelayedUnaryOperation):
    """
    A generic NDArray.

    Args:
        subdata (DelayedArray):
    """
    if delayed_base.USE_SLOTS:
        __slots__ = delayed_base.DelayedUnaryOperation.__slots__


class DelayedStack(delayed_base.DelayedNaryOperation):
    """
    Stacks multiple arrays together.
    """
    if delayed_base.USE_SLOTS:
        __slots__ = delayed_base.DelayedNaryOperation.__slots__

    def __init__(self, parts, axis):
        """
        Args:
            parts (List[DelayedArray]): data to stack
            axis (int): axes to stack on
        """
        raise NotImplementedError
        super().__init__(parts=parts)
        self.meta['axis'] = axis


class DelayedConcat(delayed_base.DelayedNaryOperation):
    """
    Stacks multiple arrays together.
    """
    if delayed_base.USE_SLOTS:
        __slots__ = delayed_base.DelayedNaryOperation.__slots__

    def __init__(self, parts, axis):
        """
        Args:
            parts (List[DelayedArray]): data to concat
            axis (int): axes to concat on
        """
        super().__init__(parts=parts)
        self.meta['axis'] = axis


class DelayedFrameStack(DelayedStack):
    """
    Stacks multiple arrays together.
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedStack.__slots__

    def __init__(self, parts):
        """
        Args:
            parts (List[DelayedArray]): data to stack
        """
        raise NotImplementedError
        super().__init__(parts=parts, axis=0)

# ------
# Images
# ------


class ImageOpsMixin:
    if delayed_base.USE_SLOTS:
        __slots__ = tuple()

    def crop(self, space_slice=None, chan_idxs=None, clip=True, wrap=True,
             pad=0, lazy=False):
        """
        Crops an image along integer pixel coordinates.

        Args:
            space_slice (Tuple[slice, slice]):
                y-slice and x-slice.

            chan_idxs (List[int]):
                indexes of bands to take

            clip (bool):
                if True, the slice is interpreted normally, where it won't go
                past the image extent, otherwise slicing into negative regions
                or past the image bounds will result in padding.  Defaults to
                True.

            wrap (bool):
                if True, negative indexes "wrap around", otherwise they are
                treated as is. Defaults to True.

            pad (int | List[Tuple[int, int]]):
                if specified, applies extra padding

            lazy (bool):
                if True, we check if the slice is equal to the image extent and
                do nothing if possible. (Introduced in 0.3.1)

        Returns:
            DelayedImage

        Example:
            >>> from delayed_image import DelayedLoad
            >>> import kwimage
            >>> self = DelayedLoad.demo().prepare()
            >>> self = self.dequantize({'quant_max': 255})
            >>> self = self.warp({'scale': 1 / 2})
            >>> pad = 0
            >>> h, w = space_dims = self.dsize[::-1]
            >>> grid = list(ub.named_product({
            >>>     'left': [0, -64], 'right': [0, 64],
            >>>     'top': [0, -64], 'bot': [0, 64],}))
            >>> grid += [
            >>>     {'left': 64, 'right': -64, 'top': 0, 'bot': 0},
            >>>     {'left': 64, 'right': 64, 'top': 0, 'bot': 0},
            >>>     {'left': 0, 'right': 0, 'top': 64, 'bot': -64},
            >>>     {'left': 64, 'right': -64, 'top': 64, 'bot': -64},
            >>> ]
            >>> crops = []
            >>> for pads in grid:
            >>>     space_slice = (slice(pads['top'], h + pads['bot']),
            >>>                    slice(pads['left'], w + pads['right']))
            >>>     delayed = self.crop(space_slice)
            >>>     crop = delayed.finalize()
            >>>     yyxx = kwimage.Boxes.from_slice(space_slice, wrap=False, clip=0).toformat('_yyxx').data[0]
            >>>     title = '[{}:{}, {}:{}]'.format(*yyxx)
            >>>     crop_canvas = kwimage.draw_header_text(crop, title, fit=True, bg_color='kw_darkgray')
            >>>     crops.append(crop_canvas)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas = kwimage.stack_images_grid(crops, pad=16, bg_value='kw_darkgreen')
            >>> canvas = kwimage.fill_nans_with_checkers(canvas)
            >>> kwplot.imshow(canvas, title='Normal Slicing: Cropped Images With Wrap+Clipped Slices', doclf=1, fnum=1)
            >>> kwplot.show_if_requested()

        Example:
            >>> # Demo the case with pads / no-clips / no-wraps
            >>> from delayed_image import DelayedLoad
            >>> import kwimage
            >>> self = DelayedLoad.demo().prepare()
            >>> self = self.dequantize({'quant_max': 255})
            >>> self = self.warp({'scale': 1 / 2})
            >>> pad = [(64, 128), (32, 96)]
            >>> pad = [(0, 20), (0, 0)]
            >>> pad = 0
            >>> pad = 8
            >>> h, w = space_dims = self.dsize[::-1]
            >>> grid = list(ub.named_product({
            >>>     'left': [0, -64], 'right': [0, 64],
            >>>     'top': [0, -64], 'bot': [0, 64],}))
            >>> grid += [
            >>>     {'left': 64, 'right': -64, 'top': 0, 'bot': 0},
            >>>     {'left': 64, 'right': 64, 'top': 0, 'bot': 0},
            >>>     {'left': 0, 'right': 0, 'top': 64, 'bot': -64},
            >>>     {'left': 64, 'right': -64, 'top': 64, 'bot': -64},
            >>> ]
            >>> crops = []
            >>> for pads in grid:
            >>>     space_slice = (slice(pads['top'], h + pads['bot']),
            >>>                    slice(pads['left'], w + pads['right']))
            >>>     delayed = self._padded_crop(space_slice, pad=pad)
            >>>     crop = delayed.finalize(optimize=1)
            >>>     yyxx = kwimage.Boxes.from_slice(space_slice, wrap=False, clip=0).toformat('_yyxx').data[0]
            >>>     title = '[{}:{}, {}:{}]'.format(*yyxx)
            >>>     if pad:
            >>>         title += f'{chr(10)}pad={pad}'
            >>>     crop_canvas = kwimage.draw_header_text(crop, title, fit=True, bg_color='kw_darkgray')
            >>>     crops.append(crop_canvas)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas = kwimage.stack_images_grid(crops, pad=16, bg_value='kw_darkgreen', resize='smaller')
            >>> canvas = kwimage.fill_nans_with_checkers(canvas)
            >>> kwplot.imshow(canvas, title='Negative Slicing: Cropped Images With clip=False wrap=False', doclf=1, fnum=2)
            >>> kwplot.show_if_requested()

        Example:
            >>> # Test lazy case
            >>> from delayed_image import DelayedLoad
            >>> self = DelayedLoad.demo().prepare()
            >>> w, h = self.dsize[0:2]
            >>> space_slice1 = (slice(0, h), slice(0, w)) # entire image
            >>> space_slice2 = (slice(None), slice(None)) # entire image
            >>> space_slice3 = (slice(0, w // 2), slice(1, h)) # subimage
            >>> result1 = self.crop(space_slice1, lazy=True)
            >>> result2 = self.crop(space_slice2, lazy=True)
            >>> result3 = self.crop(space_slice3, lazy=True)
            >>> assert result1 is self
            >>> assert result2 is self
            >>> assert result3 is not self
        """
        if lazy:
            # If we are in lazy mode and we can detect the crop wont do
            # anything then skip it.
            w, h = self.dsize
            sl_y, sl_x = space_slice
            is_noop = (
                (sl_y.start is None or sl_y.start == 0) and
                (sl_y.stop is None or sl_y.stop == h) and
                (sl_x.start is None or sl_x.start == 0) and
                (sl_x.stop is None or sl_x.stop == w) and
                (not pad)
            )
            if is_noop:
                return self

        if not clip or not wrap or pad:
            if clip or wrap:
                raise NotImplementedError(
                    ub.paragraph(
                        '''
                        Currently, in "negative slice mode" both clip and wrap
                        params must be set to False if padding is given or
                        either of clip or wrap is False.
                        '''))
            # Currently padding doesn't really work with crops, so its not
            # efficient, but we can hack it to work with warps.
            new = self._padded_crop(space_slice, pad=pad)
        else:
            # Normal efficient case
            # FIXME: This is using index-based slices and it there needs to be
            # a an explicit distinction between index and coordinate based
            # slices.
            new = DelayedCrop(self, space_slice, chan_idxs)
        return new

    def _coordinate_crop(self, roi, lazy=False):
        """
        Experimental cropping implemented as a warp, which assumes the slice in
        a coordinate-slice, and not a index-slice.

        Contextual data that needs to be known:

            * Is the box representing coordinates or indexes?

            *

        Ignore:
            >>> from delayed_image import DelayedLoad
            >>> import kwimage
            >>> raw = DelayedLoad.demo(dsize=(16, 16)).prepare()
            >>> w, h = raw.dsize[0:2]
            >>> #roi = kwimage.Box.coerce([0, 0, 8, 8], format='xywh')
            >>> roi = kwimage.Box.coerce([-.5, -.5, 8, 8], format='xywh')
            >>> roi = kwimage.Box.coerce([.5, .5, 8, 8], format='xywh')
            >>> space_slice = roi.to_slice()
            >>> result1 = raw.crop(space_slice)
            >>> result2 = raw.crop(space_slice, clip=False, wrap=False)
            >>> result3 = raw._coordinate_crop(roi, lazy=True)
            >>> result1.optimize().print_graph(fields='all')
            >>> result2.optimize().print_graph(fields='all')
            >>> result3.optimize().print_graph(fields='all')
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(raw.finalize(), pnum=(2, 1, 1), show_ticks=1, doclf=1)
            >>> roi.draw()
            >>> try:
            >>>     kwplot.imshow(result1.finalize(), pnum=(2, 3, 4), show_ticks=1, title='orig crop (index path)')
            >>> except Exception:
            >>>     kwplot.imshow(np.zeros((1, 1)), pnum=(2, 3, 4), show_ticks=1, title='orig crop (index path)')
            >>> kwplot.imshow(result2.finalize(), pnum=(2, 3, 5), show_ticks=1, title='orig crop (warp path)')
            >>> kwplot.imshow(result3.finalize(), pnum=(2, 3, 6), show_ticks=1, title='coordinate crop')
        """
        # data_dims = self.dsize[::-1]
        # _data_slice, _extra_padding = kwarray.embed_slice(
        #     space_slice, data_dims)
        tl_x = roi.tl_x
        tl_y = roi.tl_y
        coordinate_width = roi.width
        coordinate_height = roi.height
        # sl_y, sl_x = _data_slice
        # Is this correct?
        # coordinate_width = iceil(sl_x.stop - sl_x.start)
        # coordinate_height = iceil(sl_y.stop - sl_y.start)
        dsize = (coordinate_width, coordinate_height)
        transform = kwimage.Affine.translate(offset=(-tl_x, -tl_y))
        # offset1 = kwimage.Affine.translate(offset=(+.5, +.5))
        # offset2 = kwimage.Affine.translate(offset=(-.5, -.5))
        # adjusted_transform = offset2 @ transform @ offset1
        # return self.warp(adjusted_transform, dsize=dsize, interpolation='linear')
        return self.warp(transform, dsize=dsize, interpolation='linear')

    def _padded_crop(self, space_slice, pad=0):
        """
        Does the type of padded crop we want, but inefficiently using a warp.
        Reimplementing would be good, but this is good enough for now.
        """
        if self.dsize is None:
            raise Exception('dsize must be populated to do a padded crop')
        data_dims = self.dsize[::-1]
        _data_slice, _extra_padding = kwarray.embed_slice(
            space_slice, data_dims, pad)
        offset_d0, extra_d0 = _extra_padding[0]
        offset_d1, extra_d1 = _extra_padding[1]
        pad_warp = {'offset': (offset_d1, offset_d0)}
        data_crop_box = kwimage.Boxes.from_slice(
            _data_slice, clip=False, wrap=False)
        dsize = (int(data_crop_box.width.ravel()[0] + offset_d1 + extra_d1),
                 int(data_crop_box.height.ravel()[0] + offset_d0 + extra_d0))
        new = self.crop(_data_slice)
        if any([offset_d0, extra_d0, offset_d1, extra_d1]):
            # Use a warp to accomplish padding.
            # Having an explicit padding node would be better.
            new = new.warp(pad_warp, dsize=dsize)
        return new

    def warp(self, transform, dsize='auto', lazy=False, **warp_kwargs):
        """
        Applys an affine transformation to the image. See :class:`DelayedWarp`.

        Args:
            transform (ndarray | dict | kwimage.Affine):
                a coercable affine matrix.  See :class:`kwimage.Affine` for
                details on what can be coerced.

            dsize (Tuple[int, int] | str):
                The width / height of the output canvas. If 'auto', dsize is
                computed such that the positive coordinates of the warped image
                will fit in the new canvas. In this case, any pixel that maps
                to a negative coordinate will be clipped.  This has the
                property that the input transformation is not modified.

            antialias (bool):
                if True determines if the transform is downsampling and applies
                antialiasing via gaussian a blur. Defaults to False

            interpolation (str):
                interpolation code. Interpolation codes are 'linear',
                'nearest', 'cubic', 'lancsoz', and 'area'. Defaults to
                'linear'.

            border_value (int | float | str):
                if auto will be nan for float and 0 for int.

            noop_eps (float):
                This is the tolerance for optimizing a warp away.
                If the transform has all of its decomposed parameters (i.e.
                scale, rotation, translation, shear) less than this value,
                the warp node can be optimized away. Defaults to 0.

            lazy (bool):
                if True, we check if the operation would be a noop and return
                the original object instead. (Introduced in 0.3.1)

        Returns:
            DelayedImage

        Example:
            >>> from delayed_image import DelayedLoad
            >>> self = DelayedLoad.demo().prepare()
            >>> new = self.warp({'scale': 1 / 2})
            >>> assert self.dsize

        Example:
            >>> # Test with lazy
            >>> from delayed_image import DelayedLoad
            >>> self = DelayedLoad.demo().prepare()
            >>> result1 = self.warp({'scale': 1}, lazy=True)
            >>> result2 = self.warp(None, lazy=True)
            >>> result3 = self.warp(np.eye(3), lazy=True)
            >>> assert self is result1
            >>> assert self is result2
            >>> assert self is result3
        """
        if lazy:
            if transform is None:
                return self
            transform = kwimage.Affine.coerce(transform)
            if transform.isclose_identity():
                return self
        new = DelayedWarp(self, transform, dsize=dsize, **warp_kwargs)
        return new

    def scale(self, scale, dsize='auto', **warp_kwargs):
        """
        An alias for self.warp({"scale": scale}, ...)
        Backend is simply a call to :func:`ImageOpsMixin.warp`.
        """
        transform = {'scale': scale}
        return self.warp(transform, dsize=dsize, **warp_kwargs)

    def resize(self, dsize, **warp_kwargs):
        """
        Resize an image to a specific width/height by scaling it.
        Backend is simply a call to :func:`ImageOpsMixin.warp`.
        """
        old_dsize = np.array(self.dsize)
        new_dsize = np.array(dsize)
        scale = new_dsize / old_dsize
        transform = {'scale': scale}
        return self.warp(transform, dsize=dsize, **warp_kwargs)

    def dequantize(self, quantization):
        """
        Rescales image intensities from int to floats.

        Args:
            quantization (Dict[str, Any]):
                quantization information dictionary to undo.
                see :func:`delayed_image.helpers.dequantize`
                Expected keys are:
                orig_dtype (str)
                orig_min (float)
                orig_max (float)
                quant_min (float)
                quant_max (float)
                nodata (None | int)

        Returns:
            DelayedDequantize

        Example:
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> self = DelayedLoad.demo().prepare()
            >>> quantization = {
            >>>     'orig_dtype': 'float32',
            >>>     'orig_min': 0,
            >>>     'orig_max': 1,
            >>>     'quant_min': 0,
            >>>     'quant_max': 255,
            >>>     'nodata': None,
            >>> }
            >>> new = self.dequantize(quantization)
            >>> assert self.finalize().max() > 1
            >>> assert new.finalize().max() <= 1
        """
        new = DelayedDequantize(self, quantization)
        return new

    def get_overview(self, overview):
        """
        Downsamples an image by a factor of two.

        Args:
            overview (int): the overview to use (assuming it exists)

        Returns:
            DelayedOverview
        """
        new = DelayedOverview(self, overview)
        return new

    def as_xarray(self):
        """
        Returns:
            DelayedAsXarray
        """
        return DelayedAsXarray(self)

    def get_transform_from(self, src):
        """
        Find a transform from a given node (src) to this node (self / dst).

        Given two delayed images src and dst that share a common leaf, find the
        transform from src to dst.

        Args:
            src (DelayedOperation): the other view to get a transform to.
                This must share a leaf with self (which is the dst).

        Returns:
            kwimage.Affine:
                The transform that warps the space of src to the space of self.

        Example:
            >>> from delayed_image import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> base = DelayedLoad.demo().prepare()
            >>> src = base.scale(2)
            >>> dst = src.warp({'scale': 4, 'offset': (3, 5)})
            >>> transform = dst.get_transform_from(src)
            >>> tf = transform.decompose()
            >>> assert tf['scale'] == (4, 4)
            >>> assert tf['offset'] == (3, 5)

        Example:
            >>> from delayed_image import demo
            >>> self = demo.non_aligned_leafs()
            >>> leaf = list(self._leaf_paths())[0][0]
            >>> tf1 = self.get_transform_from(leaf)
            >>> tf2 = leaf.get_transform_from(self)
            >>> np.allclose(np.linalg.inv(tf2), tf1)
        """
        dst = self
        try:
            # Case where there is one known leaf
            src_from_leaf = src.get_transform_from_leaf()
            dst_from_leaf = dst.get_transform_from_leaf()
        except AttributeError:
            # This seems more robust
            src_leaf_paths = ub.udict({id(k): v for k, v in src._leaf_paths()})
            dst_leaf_paths = ub.udict({id(k): v for k, v in dst._leaf_paths()})
            common_leaf_ids = dst_leaf_paths & src_leaf_paths
            common_leaf_id = common_leaf_ids.peek_key()
            src_part = src_leaf_paths[common_leaf_id]
            dst_part = dst_leaf_paths[common_leaf_id]
            if 0:
                # In the case where we have a concatenated set of delayed images we
                # have to consider a single path from the root to a shared leaf.
                # This will work for now, but there may be a more efficient /
                # elegant way to implement it.
                src_channels = src.channels.to_oset()
                dst_channels = dst.channels.to_oset()
                common_channel = ub.peek(src_channels & dst_channels)
                src_chan = src.take_channels(common_channel)
                dst_chan = dst.take_channels(common_channel)
                src_chan = src_chan.optimize()
                dst_chan = dst_chan.optimize()
                src_part = src_chan.parts[0]
                dst_part = dst_chan.parts[0]
            src_from_leaf = src_part.get_transform_from_leaf()
            dst_from_leaf = dst_part.get_transform_from_leaf()
        dst_from_src = dst_from_leaf @ src_from_leaf.inv()
        return dst_from_src


class DelayedChannelConcat(DelayedConcat, ImageOpsMixin):
    """
    Stacks multiple arrays together.

    Example:
        >>> from delayed_image import *  # NOQA
        >>> from delayed_image.delayed_leafs import DelayedLoad
        >>> dsize = (307, 311)
        >>> c1 = DelayedNans(dsize=dsize, channels='foo')
        >>> c2 = DelayedLoad.demo('astro', dsize=dsize, channels='R|G|B').prepare()
        >>> cat = DelayedChannelConcat([c1, c2])
        >>> warped_cat = cat.warp({'scale': 1.07}, dsize=(328, 332))
        >>> warped_cat._validate()
        >>> warped_cat.finalize()

    Example:
        >>> # Test case that failed in initial implementation
        >>> # Due to incorrectly pushing channel selection under the concat
        >>> from delayed_image import *  # NOQA
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath()
        >>> base1 = DelayedLoad(fpath, channels='r|g|b').prepare()
        >>> base2 = DelayedLoad(fpath, channels='x|y|z').prepare().scale(2)
        >>> base3 = DelayedLoad(fpath, channels='i|j|k').prepare().scale(2)
        >>> bands = [base2, base1[:, :, 0].scale(2).evaluate(),
        >>>          base1[:, :, 1].evaluate().scale(2),
        >>>          base1[:, :, 2].evaluate().scale(2), base3]
        >>> delayed = DelayedChannelConcat(bands)
        >>> delayed = delayed.warp({'scale': 2})
        >>> delayed = delayed[0:100, 0:55, [0, 2, 4]]
        >>> delayed.write_network_text()
        >>> delayed.optimize()
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedConcat.__slots__ + ('dsize', 'num_channels')

    def __init__(self, parts, dsize=None):
        """
        Args:
            parts (List[DelayedArray]): data to concat
            dsize (Tuple[int, int] | None): size if known a-priori
        """
        super().__init__(parts=parts, axis=2)
        if dsize is None:
            dsize_cands = [comp.dsize for comp in self.parts]
            if not ub.allsame(dsize_cands):
                raise CoordinateCompatibilityError(
                    # 'parts must all have the same delayed size')
                    'parts must all have the same delayed size: got {}'.format(dsize_cands))
            if len(dsize_cands) == 0:
                dsize = None
            else:
                dsize = dsize_cands[0]
        self.dsize = dsize
        try:
            self.num_channels = sum(comp.num_channels for comp in self.parts)
        except TypeError:
            if any(comp.num_channels is None for comp in self.parts):
                self.num_channels = None
            else:
                raise

    def __nice__(self):
        """
        Returns:
            str
        """
        if self.channels is None:
            return '{}'.format(self.shape)
        else:
            return '{}, {}'.format(self.shape, self.channels)

    @property
    def channels(self):
        """
        Returns:
            None | FusedChannelSpec
        """
        # import delayed_image
        sub_channs = []
        for comp in self.parts:
            comp_channels = comp.channels
            if comp_channels is None:
                return None
            sub_channs.append(comp_channels)
        channs = FusedChannelSpec.concat(sub_channs)
        return channs

    @property
    def shape(self):
        """
        Returns:
            Tuple[int | None, int | None, int | None]
        """
        w, h = self.dsize
        return (h, w, self.num_channels)

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        stack = [comp._finalize() for comp in self.parts]
        if len(stack) == 1:
            final = stack[0]
        else:
            stack = [kwarray.atleast_nd(s, 3) for s in stack]
            final = np.concatenate(stack, axis=2)
        return final

    def optimize(self):
        """
        Returns:
            DelayedImage
        """
        new_parts = [part.optimize() for part in self.parts]
        kw = ub.dict_isect(self.meta, ['dsize'])
        new = self.__class__(new_parts, **kw)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('optimize DelayedChannelConcat')
        return new

    def take_channels(self, channels, missing_channel_policy='return_nan'):
        """
        This method returns a subset of the vision data with only the
        specified bands / channels.

        Args:
            channels (List[int] | slice | FusedChannelSpec):
                List of integers indexes, a slice, or a channel spec, which is
                typically a pipe (`|`) delimited list of channel codes. See
                :class:`ChannelSpec` for more detials.

            missing_channel_policy (str):
                What to do if the requested channels are missing.
                If set to 'return_nan' it will build a channel of nans which
                will allow algorithms that can handle missing data to continue.
                If set to 'error', then an ValueError will be raised.

        Returns:
            DelayedArray:
                a delayed vision operation that only operates on the following
                channels.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwcoco)
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> self = delayed = dset.coco_image(1).delay()
            >>> channels = 'B11|B8|B1|B10'
            >>> new = self.take_channels(channels)

        Example:
            >>> # xdoctest: +REQUIRES(module:kwcoco)
            >>> # Complex case
            >>> import kwcoco
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
            >>> delayed = dset.coco_image(1).delay()
            >>> astro = DelayedLoad.demo('astro', channels='r|g|b').prepare()
            >>> aligned = astro.warp(kwimage.Affine.scale(600 / 512), dsize='auto')
            >>> self = combo = DelayedChannelConcat(delayed.parts + [aligned])
            >>> channels = 'B1|r|B8|g'
            >>> new = self.take_channels(channels)
            >>> new_cropped = new.crop((slice(10, 200), slice(12, 350)))
            >>> new_opt = new_cropped.optimize()
            >>> datas = new_opt.finalize()
            >>> if 1:
            >>>     new_cropped.write_network_text(with_labels='name')
            >>>     new_opt.write_network_text(with_labels='name')
            >>> vizable = kwimage.normalize_intensity(datas, axis=2)
            >>> self._validate()
            >>> new._validate()
            >>> new_cropped._validate()
            >>> new_opt._validate()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> stacked = kwimage.stack_images(vizable.transpose(2, 0, 1))
            >>> kwplot.imshow(stacked)

        Example:
            >>> # xdoctest: +REQUIRES(module:kwcoco)
            >>> # Test case where requested channel does not exist
            >>> import kwcoco
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral', use_cache=1, verbose=100)
            >>> self = delayed = dset.coco_image(1).delay()
            >>> # by default requesting the channels that dont exist will
            >>> # return nan channels, but this can be modified by setting
            >>> # missing_channel_policy
            >>> channels = 'B1|foobar|bazbiz|B8'
            >>> new = self.take_channels(channels)
            >>> new_cropped = new.crop((slice(10, 200), slice(12, 350)))
            >>> fused = new_cropped.finalize()
            >>> assert fused.shape == (190, 338, 4)
            >>> assert np.all(np.isnan(fused[..., 1:3]))
            >>> assert not np.any(np.isnan(fused[..., 0]))
            >>> assert not np.any(np.isnan(fused[..., 3]))
            >>> # Test setting the missing channel policy
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     new = self.take_channels(channels, missing_channel_policy='error')
            >>> # test passing a bad policy
            >>> with pytest.raises(KeyError):
            >>>     new = self.take_channels(channels, missing_channel_policy='not-a-policy')
        """
        if channels is None:
            return self
        current_channels = self.channels

        if isinstance(channels, list):
            # Allows channel selection via integer indexes
            # TODO: this API needs to be tested and verified so it works
            # correctly for integers indexes and string channel names.
            top_idx_mapping = channels
            top_codes = self.channels.as_list()
            request_codes = None
        else:
            channels = FusedChannelSpec.coerce(channels)
            if current_channels == channels:
                # If the request is equal to what we already have then skip this.
                return self
            # Compute subindex integer mapping
            request_codes = channels.as_list()
            top_codes = current_channels.as_oset()
            top_idx_mapping = []
            for code in request_codes:
                try:
                    top_idx_mapping.append(top_codes.index(code))
                except KeyError:
                    top_idx_mapping.append(None)

        if missing_channel_policy == 'return_nan':
            # default behavior
            ...
        elif missing_channel_policy == 'error':
            # do a quick check for missing channels outside of the core logic
            missing_channels = [code for idx, code in zip(top_idx_mapping, request_codes)
                                if idx is None]
            if missing_channels:
                raise ValueError(
                    f'Requested channels: {missing_channels} do not exist. '
                    f'Available channels are: {current_channels}')
        else:
            raise KeyError(f'Invalid missing_channel_policy={missing_channel_policy}')

        # Rearange subcomponents into the specified channel representation
        # I am not confident that this logic is the best way to do this.
        # This may be a bottleneck.

        # This object can contain multiple subimages, which each may have
        # different numbers of channels. The FlatIndexer lets us pretend they
        # are all flattened and map from the flat index to the nested index we
        # can use to efficiently lookup the value.
        subindexer = kwarray.FlatIndexer([
            comp.num_channels for comp in self.parts])

        curr = None
        outer_accum = []
        for request_idx, idx in enumerate(top_idx_mapping):
            # If possible, keep track of the channel name this index represents
            code = None if request_codes is None else request_codes[request_idx]
            if idx is None:
                # Requested channel does not exist in our data stack
                comp = None
                inner = 0
                if curr is not None and curr.comp is None:
                    inner = curr.stop
            else:
                # Requested channel exists in our data stack
                outer, inner = subindexer.unravel(idx)
                comp = self.parts[outer]

            # Do we extend the current inner accumulator or start a new one?
            if curr is None:
                # Start the first segment
                curr = _InnerAccumSegment(comp)
            elif curr.comp is not comp:
                # accept previous segment and start a new one
                outer_accum.append(curr)
                curr = _InnerAccumSegment(comp)

            # extend this inner segment
            curr.add_inner(inner, code)

        # Accumulate final segment
        if curr is not None:
            outer_accum.append(curr)

        # Gather the subcomponents into a new delayed concat
        new_components = [curr.get_subcomponent(self.dsize) for curr in outer_accum]
        new = DelayedChannelConcat(new_components)
        return new

    def __getitem__(self, sl):
        if not isinstance(sl, tuple):
            raise TypeError('slice must be given as tuple')
        if len(sl) == 2:
            sl_y, sl_x = sl
            chan_idxs = None
        elif len(sl) == 3:
            sl_y, sl_x, chan_idxs = sl
        else:
            raise ValueError('Slice must have 2 or 3 dims')
        space_slice = (sl_y, sl_x)
        return self.crop(space_slice, chan_idxs)

    @property
    def num_overviews(self):
        """
        Returns:
            int
        """
        num_overviews = self.meta.get('num_overviews', None)
        if num_overviews is None and self.parts is not None:
            cand = [p.num_overviews for p in self.parts]
            if ub.allsame(cand):
                num_overviews = cand[0]
            else:
                import warnings
                warnings.warn('inconsistent overviews')
                num_overviews = None
        return num_overviews

    def as_xarray(self):
        """
        Returns:
            DelayedAsXarray
        """
        return DelayedAsXarray(self)

    def _push_operation_under(self, op, kwargs):
        # Note: we can't do this with a crop that has band selection
        # But spatial operations should be ok.
        new = self.__class__([op(p, **kwargs) for p in self.parts])
        if TRACE_OPTIMIZE:
            new._opt_logs.append(f'_push_operation_under {op}')
        return new

    def _validate(self):
        """
        Check that the delayed metadata corresponds with the finalized data
        """
        final = self._finalize()
        # meta_dsize = self.dsize
        meta_shape = self.shape

        final_shape = final.shape

        correspondences = {
            'shape': (final_shape, meta_shape)
        }
        for k, tup in correspondences.items():
            v1, v2 = tup
            if v1 != v2:
                raise AssertionError(
                    f'final and meta {k} does not agree {v1!r} != {v2!r}')
        return self

    def undo_warps(self, remove=None, retain=None, squash_nans=False, return_warps=False):
        """
        Attempts to "undo" warping for each concatenated channel and returns a
        list of delayed operations that are cropped to the right regions.

        Typically you will retrain offset, theta, and shear to remove scale.
        This ensures the data is spatially aligned up to a scale factor.

        Args:
            remove (List[str]): if specified, list components of the warping to
                remove. Can include: "offset", "scale", "shearx", "theta".
                Typically set this to ["scale"].

            retain (List[str]): if specified, list components of the warping to
                retain. Can include: "offset", "scale", "shearx", "theta".
                Mutually exclusive with "remove". If neither remove or retain
                is specified, retain is set to ``[]``.

            squash_nans (bool):
                if True, pure nan channels are squashed into a 1x1 array as
                they do not correspond to a real source.

            return_warps (bool):
                if True, return the transforms we applied. I.e. the transform
                from the ``self`` to the returned ``parts``.
                This is useful when you need to warp objects in the original
                space into the jagged space.

        Returns:
            List[DelayedImage] | Tuple[List[DelayedImage] | List[kwimage.Affine]]:
                The List[DelayedImage] are the ``parts`` i.e. the new images with the warping undone.
                The List[kwimage.Affine]: is the transforms from ``self`` to each item in ``parts``

        Note:
            The most common use case is to get aligned images, but at their
            native scale, to do this use the argument: ``remove=['scale']``.

        Example:
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> from delayed_image.delayed_leafs import DelayedNans
            >>> import ubelt as ub
            >>> import kwimage
            >>> import kwarray
            >>> import numpy as np
            >>> # Demo case where we have different channels at different resolutions
            >>> base = DelayedLoad.demo(channels='r|g|b').prepare().dequantize({'quant_max': 255})
            >>> bandR = base[:, :, 0].scale(100 / 512)[:, :-50].evaluate()
            >>> bandG = base[:, :, 1].scale(300 / 512).warp({'theta': np.pi / 8, 'about': (150, 150)}).evaluate()
            >>> bandB = base[:, :, 2].scale(600 / 512)[:150, :].evaluate()
            >>> bandN = DelayedNans((600, 600), channels='N')
            >>> # Make a concatenation of images of different underlying native resolutions
            >>> delayed_vidspace = DelayedChannelConcat([
            >>>     bandR.scale(6, dsize=(600, 600)).optimize(),
            >>>     bandG.warp({'theta': -np.pi / 8, 'about': (150, 150)}).scale(2, dsize=(600, 600)).optimize(),
            >>>     bandB.scale(1, dsize=(600, 600)).optimize(),
            >>>     bandN,
            >>> ]).warp({'scale': 0.7}).optimize()
            >>> vidspace_box = kwimage.Boxes([[100, 10, 270, 160]], 'ltrb')
            >>> vidspace_poly = vidspace_box.to_polygons()[0]
            >>> vidspace_slice = vidspace_box.to_slices()[0]
            >>> self = delayed_vidspace[vidspace_slice].optimize()
            >>> print('--- Aligned --- ')
            >>> self.write_network_text()
            >>> squash_nans = True
            >>> undone_all_parts, tfs1 = self.undo_warps(squash_nans=squash_nans, return_warps=True)
            >>> undone_scale_parts, tfs2 = self.undo_warps(remove=['scale'], squash_nans=squash_nans, return_warps=True)
            >>> stackable_aligned = self.finalize().transpose(2, 0, 1)
            >>> stackable_undone_all = []
            >>> stackable_undone_scale = []
            >>> print('--- Undone All --- ')
            >>> for undone in undone_all_parts:
            ...     undone.write_network_text()
            ...     stackable_undone_all.append(undone.finalize())
            >>> print('--- Undone Scale --- ')
            >>> for undone in undone_scale_parts:
            ...     undone.write_network_text()
            ...     stackable_undone_scale.append(undone.finalize())
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas0 = kwimage.stack_images(stackable_aligned, axis=1)
            >>> canvas1 = kwimage.stack_images(stackable_undone_all, axis=1)
            >>> canvas2 = kwimage.stack_images(stackable_undone_scale, axis=1)
            >>> canvas0 = kwimage.draw_header_text(canvas0, 'Rescaled Aligned Channels')
            >>> canvas1 = kwimage.draw_header_text(canvas1, 'Unwarped Channels')
            >>> canvas2 = kwimage.draw_header_text(canvas2, 'Unscaled Channels')
            >>> canvas = kwimage.stack_images([canvas0, canvas1, canvas2], axis=0)
            >>> canvas = kwimage.fill_nans_with_checkers(canvas)
            >>> kwplot.imshow(canvas)
        """
        retain = _rectify_retain(remove, retain)
        unwarped_parts = []
        if return_warps:
            jagged_warps = []
            for part in self.parts:
                undone_part, undo_warp = part.undo_warp(
                    retain=retain, squash_nans=squash_nans,
                    return_warp=return_warps)
                unwarped_parts.append(undone_part)
                jagged_warps.append(undo_warp)
            return unwarped_parts, jagged_warps
        else:
            for part in self.parts:
                undone_part = part.undo_warp(
                    retain=retain, squash_nans=squash_nans)
                unwarped_parts.append(undone_part)
            return unwarped_parts


class DelayedImage(DelayedArray, ImageOpsMixin):
    """
    For the case where an array represents a 2D image with multiple channels
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedArray.__slots__

    def __init__(self, subdata=None, dsize=None, channels=None):
        """
        Args:
            subdata (DelayedArray):
            dsize (None | Tuple[int | None, int | None]): overrides subdata dsize
            channels (None | int | FusedChannelSpec): overrides subdata channels
        """
        super().__init__(subdata)
        if channels is None:
            num_channels = None
        else:
            if isinstance(channels, int):
                num_channels = channels
                channels = None
            else:
                channels = FusedChannelSpec.coerce(channels)
                num_channels = channels.normalize().numel()
        self.meta['channels'] = channels
        self.meta['num_channels'] = num_channels

        self.meta['dsize'] = dsize

    def __nice__(self):
        """
        Returns:
            str
        """
        if self.channels is None:
            return '{}'.format(self.shape)
        else:
            return '{}, {}'.format(self.shape, self.channels)

    @property
    def shape(self):
        """
        Returns:
            None | Tuple[int | None, int | None, int | None]
        """
        dsize = self.dsize
        if dsize is None:
            dsize = (None, None)
        w, h = dsize
        c = self.num_channels
        return (h, w, c)

    @property
    def num_channels(self):
        """
        Returns:
            None | int
        """
        num_channels = self.meta.get('num_channels', None)
        if num_channels is None and self.subdata is not None:
            num_channels = self.subdata.num_channels
        return num_channels

    @property
    def dsize(self):
        """
        Returns:
            None | Tuple[int | None, int | None]
        """
        # return self.meta.get('dsize', None)
        dsize = self.meta.get('dsize', None)
        if dsize is None and self.subdata is not None:
            dsize = self.subdata.dsize
        return dsize

    @property
    def channels(self):
        """
        Returns:
            None | FusedChannelSpec
        """
        channels = self.meta.get('channels', None)
        if channels is None and self.subdata is not None:
            channels = self.subdata.channels
        return channels

    @channels.setter
    def channels(self, channels):
        if channels is None:
            num_channels = None
        else:
            if isinstance(channels, int):
                num_channels = channels
                channels = None
            else:
                channels = FusedChannelSpec.coerce(channels)
                num_channels = channels.normalize().numel()
        self.meta['channels'] = channels
        self.meta['num_channels'] = num_channels

    @property
    def num_overviews(self):
        """
        Returns:
            int
        """
        num_overviews = self.meta.get('num_overviews', None)
        if num_overviews is None and self.subdata is not None:
            num_overviews = self.subdata.num_overviews
        return num_overviews

    def __getitem__(self, sl):
        if not isinstance(sl, tuple):
            raise TypeError('slice must be given as tuple')
        if len(sl) == 2:
            sl_y, sl_x = sl
            chan_idxs = None
        elif len(sl) == 3:
            sl_y, sl_x, chan_idxs = sl
        else:
            raise ValueError('Slice must have 2 or 3 dims')
        space_slice = (sl_y, sl_x)
        return self.crop(space_slice, chan_idxs)

    def take_channels(self, channels, lazy=False,
                      missing_channel_policy='return_nan'):
        """
        This method returns a subset of the vision data with only the
        specified bands / channels.

        Args:
            channels (List[int] | slice | FusedChannelSpec):
                List of integers indexes, a slice, or a channel spec, which is
                typically a pipe (`|`) delimited list of channel codes. See
                :class:`ChannelSpec` for more detials.

            lazy (bool):
                if True, dont create a new object if we can detect that it
                would be a noop.

            missing_channel_policy (str):
                What to do if the requested channels are missing.
                If set to 'return_nan' it will build a channel of nans which
                will allow algorithms that can handle missing data to continue.
                If set to 'error', then an ValueError will be raised.

        Returns:
            DelayedCrop:
                a new delayed load with a fused take channel operation

        Note:
            The channel subset must exist here or it will raise an error.
            A better implementation (via pymbolic) might be able to do better

        Example:
            >>> #
            >>> # Test Channel Select Via Code
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image import DelayedLoad
            >>> self = DelayedLoad.demo(dsize=(16, 16), channels='r|g|b').prepare()
            >>> channels = 'r|b'
            >>> new = self.take_channels(channels)._validate()
            >>> new2 = new[:, :, [1, 0]]._validate()
            >>> new3 = new2[:, :, [1]]._validate()

        Example:
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image import DelayedLoad
            >>> self = DelayedLoad.demo('astro').prepare()
            >>> channels = [2, 0]
            >>> new = self.take_channels(channels)
            >>> new3 = new.take_channels([1, 0])
            >>> new._validate()
            >>> new3._validate()

            >>> final1 = self.finalize()
            >>> final2 = new.finalize()
            >>> final3 = new3.finalize()
            >>> assert np.all(final1[..., 2] == final2[..., 0])
            >>> assert np.all(final1[..., 0] == final2[..., 1])
            >>> assert final2.shape[2] == 2

            >>> assert np.all(final1[..., 2] == final3[..., 1])
            >>> assert np.all(final1[..., 0] == final3[..., 0])
            >>> assert final3.shape[2] == 2

        Example:
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image import DelayedLoad
            >>> self = DelayedLoad.demo(dsize=(16, 16), channels='r|g|b').prepare()
            >>> # Case where a channel doesn't exist
            >>> channels = 'r|b|magic'
            >>> new = self.take_channels(channels)
            >>> assert len(new.parts) == 2
            >>> new._validate()
        """
        if isinstance(channels, list):
            top_idx_mapping = channels
        else:
            if channels is None and lazy:
                return self
            current_channels = self.channels
            if current_channels is None:
                raise ValueError(
                    'The channel spec for this node are unknown. '
                    'Cannot use a spec to select them'
                )
            channels = FusedChannelSpec.coerce(channels)
            if lazy and current_channels == channels:
                return self
            # Computer subindex integer mapping
            request_codes = channels.as_list()
            top_codes = current_channels.as_oset()
            try:
                top_idx_mapping = [
                    top_codes.index(code)
                    for code in request_codes
                ]
            except KeyError:
                # If a requested channel doesn't exist we break this node up
                # into a concat node so we can use its nan handing logic
                # This should be easy to optimize if necessary.
                wrp = DelayedChannelConcat([self], dsize=self.dsize)
                new =  wrp.take_channels(
                    channels, missing_channel_policy=missing_channel_policy)
                return new
        new_chan_ixs = top_idx_mapping
        new = self.crop(None, new_chan_ixs)
        return new

    def _validate(self):
        """
        Check that the delayed metadata corresponds with the finalized data
        """
        opt = self.optimize()
        opt_shape = opt.shape

        final = self._finalize()
        # meta_dsize = self.dsize
        meta_shape = self.shape

        final_shape = final.shape

        correspondences = {
            'opt_chans': (self.channels, opt.channels),
            'opt_nbands': (self.num_channels, opt.num_channels),
            'final_shape': (final_shape, meta_shape),
            'opt_shape': (opt_shape, meta_shape),
        }
        for k, tup in correspondences.items():
            v1, v2 = tup
            if v1 != v2:
                raise AssertionError(
                    f'final and meta {k} does not agree {v1!r} != {v2!r}')
        return self

    def _transform_from_subdata(self):
        raise NotImplementedError

    def get_transform_from_leaf(self):
        """
        Returns the transformation that would align data with the leaf
        """
        subdata_from_leaf = self.subdata.get_transform_from_leaf()
        self_from_subdata = self._transform_from_subdata()
        self_from_leaf = self_from_subdata @ subdata_from_leaf
        return self_from_leaf

    def evaluate(self):
        """
        Evaluate this node and return the data as an identity.

        Returns:
            DelayedIdentity
        """
        final = self.finalize()
        new = delayed_leafs.DelayedIdentity(final, dsize=self.dsize,
                                            channels=self.channels)
        return new

    def _opt_push_under_concat(self):
        """
        Push this node under its child node if it is a concatenation operation
        """
        assert isinstance2(self.subdata, DelayedChannelConcat)
        kwargs = ub.compatible(self.meta, self.__class__.__init__)
        new = self.subdata._push_operation_under(self.__class__, kwargs)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('_opt_push_under_concat')
        return new

    def undo_warp(self, remove=None, retain=None, squash_nans=False, return_warp=False):
        """
        Attempts to "undo" warping for each concatenated channel and returns a
        list of delayed operations that are cropped to the right regions.

        Typically you will retrain offset, theta, and shear to remove scale.
        This ensures the data is spatially aligned up to a scale factor.

        Args:
            remove (List[str]): if specified, list components of the warping to
                remove. Can include: "offset", "scale", "shearx", "theta".
                Typically set this to ["scale"].

            retain (List[str]): if specified, list components of the warping to
                retain. Can include: "offset", "scale", "shearx", "theta".
                Mutually exclusive with "remove". If neither remove or retain
                is specified, retain is set to ``[]``.

            squash_nans (bool):
                if True, pure nan channels are squashed into a 1x1 array as
                they do not correspond to a real source.

            return_warp (bool):
                if True, return the transform we applied.
                This is useful when you need to warp objects in the original
                space into the jagged space.

        SeeAlso:
            DelayedChannelConcat.undo_warps

        Example:
            >>> # Test similar to undo_warps, but on each channel separately
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> from delayed_image.delayed_leafs import DelayedNans
            >>> import ubelt as ub
            >>> import kwimage
            >>> import kwarray
            >>> import numpy as np
            >>> # Demo case where we have different channels at different resolutions
            >>> base = DelayedLoad.demo(channels='r|g|b').prepare().dequantize({'quant_max': 255})
            >>> bandR = base[:, :, 0].scale(100 / 512)[:, :-50].evaluate()
            >>> bandG = base[:, :, 1].scale(300 / 512).warp({'theta': np.pi / 8, 'about': (150, 150)}).evaluate()
            >>> bandB = base[:, :, 2].scale(600 / 512)[:150, :].evaluate()
            >>> bandN = DelayedNans((600, 600), channels='N')
            >>> B0 = bandR.scale(6, dsize=(600, 600)).optimize()
            >>> B1 = bandG.warp({'theta': -np.pi / 8, 'about': (150, 150)}).scale(2, dsize=(600, 600)).optimize()
            >>> B2 = bandB.scale(1, dsize=(600, 600)).optimize()
            >>> vidspace_box = kwimage.Boxes([[-10, -10, 270, 160]], 'ltrb').scale(1 / .7).quantize()
            >>> vidspace_poly = vidspace_box.to_polygons()[0]
            >>> vidspace_slice = vidspace_box.to_slices()[0]
            >>> # Test with the padded crop
            >>> self0 = B0.crop(vidspace_slice, wrap=0, clip=0, pad=10).optimize()
            >>> self1 = B1.crop(vidspace_slice, wrap=0, clip=0, pad=10).optimize()
            >>> self2 = B2.crop(vidspace_slice, wrap=0, clip=0, pad=10).optimize()
            >>> parts = [self0, self1, self2]
            >>> # Run the undo on each channel
            >>> undone_scale_parts = [d.undo_warp(remove=['scale']) for d in parts]
            >>> print('--- Aligned --- ')
            >>> stackable_aligned = []
            >>> for d in parts:
            >>>     d.write_network_text()
            >>>     stackable_aligned.append(d.finalize())
            >>> print('--- Undone Scale --- ')
            >>> stackable_undone_scale = []
            >>> for undone in undone_scale_parts:
            ...     undone.write_network_text()
            ...     stackable_undone_scale.append(undone.finalize())
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas0 = kwimage.stack_images(stackable_aligned, axis=1, pad=5, bg_value='kw_darkgray')
            >>> canvas2 = kwimage.stack_images(stackable_undone_scale, axis=1, pad=5, bg_value='kw_darkgray')
            >>> canvas0 = kwimage.draw_header_text(canvas0, 'Rescaled Channels')
            >>> canvas2 = kwimage.draw_header_text(canvas2, 'Native Scale Channels')
            >>> canvas = kwimage.stack_images([canvas0, canvas2], axis=0, bg_value='kw_darkgray')
            >>> canvas = kwimage.fill_nans_with_checkers(canvas)
            >>> kwplot.imshow(canvas)
        """
        retain = _rectify_retain(remove, retain)
        part = self
        tf_root_from_leaf = part.get_transform_from_leaf()
        tf_leaf_from_root = tf_root_from_leaf.inv()
        undo_all = tf_leaf_from_root
        all_undo_components = undo_all.concise()
        undo_components = ub.dict_diff(all_undo_components, retain)
        undo_warp = kwimage.Affine.coerce(undo_components)
        undone_part = part.warp(undo_warp).optimize()
        if squash_nans:
            if return_warp:
                # hack the return undo_warp
                w, h = undone_part.dsize
                undo_warp = kwimage.Affine.scale((1 / w, 1 / h)) @ undo_warp
            if isinstance2(undone_part, delayed_leafs.DelayedNans):
                undone_part = undone_part[0:1, 0:1].optimize()
        if return_warp:
            return undone_part, undo_warp
        else:
            return undone_part


class DelayedAsXarray(DelayedImage):
    """
    Casts the data to an xarray object in the finalize step

    Example;
        >>> # xdoctest: +REQUIRES(module:xarray)
        >>> from delayed_image.delayed_nodes import *  # NOQA
        >>> from delayed_image import DelayedLoad
        >>> # without channels
        >>> base = DelayedLoad.demo(dsize=(16, 16)).prepare()
        >>> self = base.as_xarray()
        >>> final = self._validate().finalize()
        >>> assert len(final.coords) == 0
        >>> assert final.dims == ('y', 'x', 'c')
        >>> # with channels
        >>> base = DelayedLoad.demo(dsize=(16, 16), channels='r|g|b').prepare()
        >>> self = base.as_xarray()
        >>> final = self._validate().finalize()
        >>> assert final.coords.indexes['c'].tolist() == ['r', 'g', 'b']
        >>> assert final.dims == ('y', 'x', 'c')
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedImage.__slots__

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        import xarray as xr
        subfinal = np.asarray(self.subdata._finalize())
        channels = self.subdata.channels
        coords = {}
        if channels is not None:
            coords['c'] = channels.code_list()
            if len(subfinal.shape) == 2:
                subfinal = subfinal[:, :, None]
        final = xr.DataArray(subfinal, dims=('y', 'x', 'c'), coords=coords)
        return final

    def optimize(self):
        """
        Returns:
            DelayedImage
        """
        new = self.subdata.optimize().as_xarray()
        if TRACE_OPTIMIZE:
            new._opt_logs.append('optimize DelayedAsXarray')
        return new


class DelayedWarp(DelayedImage):
    """
    Applies an affine transform to an image.

    Example:
        >>> from delayed_image.delayed_nodes import *  # NOQA
        >>> from delayed_image import DelayedLoad
        >>> self = DelayedLoad.demo(dsize=(16, 16)).prepare()
        >>> warp1 = self.warp({'scale': 3})
        >>> warp2 = warp1.warp({'theta': 0.1})
        >>> warp3 = warp2._opt_fuse_warps()
        >>> warp3._validate()
        >>> print(ub.urepr(warp2.nesting(), nl=-1, sort=0))
        >>> print(ub.urepr(warp3.nesting(), nl=-1, sort=0))
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedImage.__slots__ + ('_data_keys', '_algo_keys')

    def __init__(self, subdata, transform, dsize='auto', antialias=True,
                 interpolation='linear', border_value='auto', noop_eps=0,
                 backend='auto'):
        """
        Args:
            subdata (DelayedArray): data to operate on

            transform (ndarray | dict | kwimage.Affine):
                a coercable affine matrix.  See :class:`kwimage.Affine` for
                details on what can be coerced.

            dsize (Tuple[int, int] | str):
                The width / height of the output canvas. If 'auto', dsize is
                computed such that the positive coordinates of the warped image
                will fit in the new canvas. In this case, any pixel that maps
                to a negative coordinate will be clipped.  This has the
                property that the input transformation is not modified.

            antialias (bool):
                if True determines if the transform is downsampling and applies
                antialiasing via gaussian a blur. Defaults to False

            interpolation (str):
                interpolation code or cv2 integer. Interpolation codes are linear,
                nearest, cubic, lancsoz, and area. Defaults to "linear".

            noop_eps (float):
                This is the tolerance for optimizing a warp away.
                If the transform has all of its decomposed parameters (i.e.
                scale, rotation, translation, shear) less than this value,
                the warp node can be optimized away. Defaults to 0.
        """
        super().__init__(subdata)
        transform = kwimage.Affine.coerce(transform)
        if dsize == 'auto':
            from delayed_image.helpers import _auto_dsize
            dsize = _auto_dsize(transform, self.subdata.dsize)
        self.meta['transform'] = transform
        self.meta['dsize'] = dsize
        self.meta['antialias'] = antialias
        self.meta['interpolation'] = interpolation
        self.meta['border_value'] = border_value
        self.meta['noop_eps'] = noop_eps
        self.meta['backend'] = backend
        # Mark which keys need to be passed around and for what reason
        self._data_keys = ['transform', 'dsize']
        self._algo_keys = [
            'interpolation', 'antialias', 'border_value', 'noop_eps']

    @property
    def transform(self):
        """
        Returns:
            kwimage.Affine
        """
        return self.meta['transform']

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        dsize = self.dsize
        if dsize == (None, None):
            dsize = None
        antialias = self.meta['antialias']
        transform = self.meta['transform']
        interpolation = self.meta['interpolation']
        backend = self.meta['backend']

        prewarp = self.subdata._finalize()
        prewarp = np.asanyarray(prewarp)

        # TODO: we could configure this, but forcing nans on floats seems like
        # a pretty nice default border behavior. It would be even nicer to have
        # masked arrays for ints.
        # The scalar / explicit functionality will be handled inside warp_affine
        # in the future, so some of this can be removed.
        num_chan = kwimage.num_channels(prewarp)
        if self.meta['border_value'] == 'auto':
            if prewarp.dtype.kind == 'f':
                border_value = np.nan
            else:
                if isinstance(prewarp, np.ma.MaskedArray):
                    border_value = int(prewarp.fill_value)
                else:
                    border_value = 0
        else:
            border_value = self.meta['border_value']

        if not isinstance(border_value, str):
            if not ub.iterable(border_value):
                # Odd OpenCV behavior: https://github.com/opencv/opencv/issues/22283
                # Can only have at most 4 components to border_value and
                # then they start to wrap around. This is fine if we are only
                # specifying a single number for all channels
                border_value = (border_value,) * min(4, num_chan)
            if len(border_value) > 4:
                raise ValueError('borderValue cannot have more than 4 components. '
                                 'OpenCV #22283 describes why')

            # HACK:
            # the border value only correctly applies to the first 4 channels for
            # whatever reason.
            border_value = border_value[0:4]

        from delayed_image.helpers import _ensure_valid_dsize
        dsize = _ensure_valid_dsize(dsize)

        M = np.asarray(transform)
        final = kwimage.warp_affine(prewarp, M, dsize=dsize,
                                    interpolation=interpolation,
                                    antialias=antialias,
                                    border_value=border_value,
                                    origin_convention='corner',
                                    backend=backend,
                                    )
        # final = kwimage.warp_projective(sub_data_, M, dsize=dsize, flags=flags)
        # Ensure that the last dimension is channels
        final = kwarray.atleast_nd(final, 3, front=False)
        return final

    def optimize(self):
        """
        Returns:
            DelayedImage

        Example:
            >>> # Demo optimization that removes a noop warp
            >>> from delayed_image import DelayedLoad
            >>> import kwimage
            >>> base = DelayedLoad.demo(channels='r|g|b').prepare()
            >>> self = base.warp(kwimage.Affine.eye())
            >>> new = self.optimize()
            >>> assert len(self.as_graph().nodes) == 2
            >>> assert len(new.as_graph().nodes) == 1

        Example:
            >>> # Test optimize nans
            >>> from delayed_image import DelayedNans
            >>> import kwimage
            >>> base = DelayedNans(dsize=(100, 100), channels='a|b|c')
            >>> self = base.warp(kwimage.Affine.scale(0.1))
            >>> # Should simply return a new nan generator
            >>> new = self.optimize()
            >>> assert len(new.as_graph().nodes) == 1

        Example:
            >>> # Test optimize nans
            >>> from delayed_image import DelayedLoad
            >>> import kwimage
            >>> base = DelayedLoad.demo(channels='r|g|b').prepare()
            >>> transform = kwimage.Affine.scale(1.0 + 1e-7)
            >>> self = base.warp(transform, dsize=base.dsize)
            >>> # An optimize will not remove a warp if there is any
            >>> # doubt if it is the identity.
            >>> new = self.optimize()
            >>> assert len(self.as_graph().nodes) == 2
            >>> assert len(new.as_graph().nodes) == 2
            >>> # But we can specify a threshold where it will
            >>> self._set_nested_params(noop_eps=1e-6)
            >>> new = self.optimize()
            >>> assert len(self.as_graph().nodes) == 2
            >>> assert len(new.as_graph().nodes) == 1
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance2(new.subdata, DelayedWarp):
            new = new._opt_fuse_warps()

        # Check if the transform is close enough to identity to be considered
        # negligable.
        noop_eps = new.meta['noop_eps']
        is_negligable = (
            new.dsize == new.subdata.dsize and
            new.transform.isclose_identity(rtol=noop_eps, atol=noop_eps)
        )
        if is_negligable:
            new = new.subdata
            if TRACE_OPTIMIZE:
                new._opt_logs.append('Contract identity warp')
        elif isinstance2(new.subdata, DelayedChannelConcat):
            new = new._opt_push_under_concat().optimize()
        elif hasattr(new.subdata, '_optimized_warp'):
            # The subdata knows how to optimize itself wrt a warp
            warp_kwargs = ub.dict_isect(
                self.meta, self._data_keys + self._algo_keys)
            new = new.subdata._optimized_warp(**warp_kwargs).optimize()
        else:
            split = new._opt_split_warp_overview()
            if new is not split:
                new = split
                new.subdata = new.subdata.optimize()
                new = new.optimize()
            else:
                new = new._opt_absorb_overview()
        if TRACE_OPTIMIZE:
            new._opt_logs.append('optimize DelayedWarp')
        return new

    def _transform_from_subdata(self):
        return self.transform

    def _opt_fuse_warps(self):
        """
        Combine two consecutive warps into a single operation.
        """
        assert isinstance2(self.subdata, DelayedWarp)

        DEBUG = 0
        if DEBUG:
            print('before fuse warps')
            self.print_graph()

        inner_data = self.subdata.subdata
        tf1 = self.subdata.meta['transform']
        tf2 = self.meta['transform']
        # TODO: could ensure the metadata is compatable, for now just take the
        # most recent
        dsize = self.meta['dsize']
        common_meta = ub.dict_isect(self.meta, self._algo_keys)
        new_transform = tf2 @ tf1
        new = self.__class__(inner_data, new_transform, dsize=dsize,
                             **common_meta)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('Fuse warps')

        if DEBUG:
            print('after fuse warps')
            new.print_graph()
        return new

    def _opt_absorb_overview(self):
        """
        Remove any deeper overviews that would be undone by this warp.

        Given this warp node, if it has a scale component could undo an
        overview (i.e. the scale factor is greater than 2), we want to:

            1. determine if there is an overview deeper in the tree.
            2. remove that overview and that scale factor from this warp
            3. modify any intermediate nodes that will be changed by having the
               deeper overview removed.

        NOTE:
            This optimization is currently the most dubious one in the code,
            and is likely where some of the bugs are coming from.  Help wanted.

        CommandLine:
           xdoctest -m delayed_image.delayed_nodes DelayedWarp._opt_absorb_overview

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image import DelayedLoad
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> base = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> # Case without any operations between the overview and warp
            >>> self = base.get_overview(1).warp({'scale': 4})
            >>> self.write_network_text()
            >>> opt = self._opt_absorb_overview()._validate()
            >>> opt.write_network_text()
            >>> opt_data = [d for n, d in opt.as_graph().nodes(data=True)]
            >>> assert 'DelayedOverview' not in [d['type'] for d in opt_data]
            >>> # Case with a chain of operations between overview and warp
            >>> self = base.get_overview(1)[0:101, 0:100].warp({'scale': 4})
            >>> self.write_network_text()
            >>> opt = self._opt_absorb_overview()._validate()
            >>> opt.write_network_text()
            >>> opt_data = [d for n, d in opt.as_graph().nodes(data=True)]
            >>> #assert opt_data[1]['meta']['space_slice'] == (slice(0, 202, None), slice(0, 200, None))
            >>> assert opt_data[1]['meta']['space_slice'] == (slice(0, 204, None), slice(0, 202, None))
            >>> # Any sort of complex chain does prevents this optimization
            >>> # from running.
            >>> self = base.get_overview(1)[0:101, 0:100][0:50, 0:50].warp({'scale': 4})
            >>> opt = self._opt_absorb_overview()._validate()
            >>> opt.write_network_text()
            >>> opt_data = [d for n, d in opt.as_graph().nodes(data=True)]
            >>> assert 'DelayedOverview' in [d['type'] for d in opt_data]
        """
        DEBUG = 0

        # Check if there is a strict downsampling component
        transform = self.meta['transform']

        # Old Slow Code:
        # params = transform.decompose()
        # sx, sy = params['scale']
        # New Optimized Code:
        from delayed_image.helpers import _decompose_scale
        sx, sy = _decompose_scale(transform)

        eps = 1e-8
        twoish = (2 - eps)
        if sx < twoish and sy < twoish:
            return self

        # Lookahead to see if there is a nearby overview operation that can be
        # absorbed, remember the chain of operations between the warp and the
        # overview, as it will need to be modified.

        # !!! FIXME !!!
        # The number of nodes we lookahead is hard coded based on reasonable
        # cases we expect in the real world. A correct implementation would not
        # depend on a hard coded number like this. However, this optimization
        # may be able to be completely refactored to avoid this approach all
        # together.
        LOOKAHEAD_NUMBER = 4

        parent = self
        subdata = None
        chain = []
        num_dcrops = 0
        for i in range(LOOKAHEAD_NUMBER):
            subdata = parent.subdata
            if subdata is None:
                break
            elif isinstance2(subdata, DelayedWarp):
                subdata = None
                break
            elif isinstance2(subdata, DelayedOverview):
                # We found an overview node
                overview = subdata
                break
            elif isinstance2(subdata, DelayedDequantize):
                pass
            elif isinstance2(subdata, DelayedCrop):
                num_dcrops += 1
            else:
                subdata = None
                break
            chain.append(subdata)
            parent = subdata
        else:
            subdata = None

        if subdata is None:
            return self

        if num_dcrops > 1:
            # The following logic does not work there is more than a single
            # crop between the warp and the overview. Punt and just return.
            # Might be better to fuse the crops.
            return self

        if DEBUG:
            print('---------')
            print('ORIG:')
            self.print_graph()

        # At this point we have some chain:
        # [Warp, Something, ..., Overview]
        # The chain is the [Something, ...] part

        # Replace the overview node with a warp node that mimics it.
        # This has no impact on the function of the operation stack.
        mimic_overview = overview._opt_overview_as_warp()
        tf1 = mimic_overview.meta['transform']

        # Handle any nodes between the warp and the overview.
        # This should be at most one quantization and one crop operation,
        # but we may generalize that in the future.

        if DEBUG:
            print('chain = {}'.format(ub.urepr(chain, nl=1)))

        if not chain:
            # The overview is directly after this warp
            new_head = mimic_overview.subdata
            if TRACE_OPTIMIZE:
                new_head._opt_logs.extend(mimic_overview._opt_logs)
                new_head._opt_logs.append('_opt_absorb_overview:absorb_neighbor')
        else:
            # Copy the chain so this does not mutate the input
            chain = [copy.copy(n) for n in chain]
            for u, v in ub.iter_window(chain, 2):
                u.subdata = v
            tail = chain[-1]
            if TRACE_OPTIMIZE:
                mimic_overview._opt_logs.extend(tail.subdata._opt_logs)
            tail.subdata = mimic_overview
            # Check if the tail of the chain is a crop.
            if hasattr(tail, '_opt_warp_after_crop'):
                # This modifies the tail so it is now a warp followed by a
                # crop. Note that the warp may be different than the mimicked
                # overview, so use this new transform instead.
                # (Actually, I think this can't make the new crop non integral,
                # so it probably wont matter)
                if DEBUG:
                    print('HAS CROP OP')
                    print('Tail')
                    tail.print_graph()
                modified_tail = tail._opt_warp_after_crop()
                if DEBUG:
                    print('Modified Tail')
                    modified_tail.print_graph()
                    print('modified_tail = {}'.format(ub.urepr(modified_tail, nl=1)))

                # Previous code used the "modified tail" to grab the new dsize
                # that will be applied to the rest of the chain. However, the
                # modified tail corresponds to the warp we are going to remove.
                # We need to grab it from the modified crop instead, which is
                # going to be the child of the modified tail.

                # Thus we comment out this line:
                # new_chain_dsize = modified_tail.meta['dsize']

                # And use this as the fixed way to grab the dsize.  Tests pass,
                # but I'm leaving this note in here in case this change causes
                # unforeseen issues.
                new_chain_dsize = modified_tail.subdata.meta['dsize']

                tf1 = modified_tail.meta['transform']
                # Remove the modified warp
                tail_parent = chain[-2] if len(chain) > 1 else self
                new_tail = modified_tail.subdata

                if DEBUG:
                    print('tail_parent = {}'.format(ub.urepr(tail_parent, nl=1)))
                    print('new_tail = {}'.format(ub.urepr(new_tail, nl=1)))
                if TRACE_OPTIMIZE:
                    new_tail._opt_logs.extend(modified_tail._opt_logs)
                    new_tail._opt_logs.extend(tail_parent.subdata._opt_logs)
                    new_tail._opt_logs.append('_opt_absorb_overview:modify-tail')
                tail_parent.subdata = new_tail
                chain[-1] = new_tail
                for notcrop in chain[:-1]:
                    notcrop.meta['dsize'] = new_chain_dsize
            else:
                # The chain does not contain a crop operation, we can safely
                # remove it. Finally remove the overview transform entirely
                if DEBUG:
                    print('NO CROP OP')
                tail.subdata = mimic_overview.subdata
                new_chain_dsize = mimic_overview.subdata.meta['dsize']
                for notcrop in chain:
                    notcrop.meta['dsize'] = new_chain_dsize
                if TRACE_OPTIMIZE:
                    tail._opt_logs.extend(mimic_overview._opt_logs)
                    tail._opt_logs.append('_opt_absorb_overview:safe-to-remove')

            # The dsize within the chain might be wrong due to our
            # modification. I **think** its ok to just directly set it to the
            # new dsize as it should only be operations that do not change the
            # dsize, but it would be nice to find a more elegant
            # implementation.
            if DEBUG:
                print(f'new_chain_dsize={new_chain_dsize}')
            for notcrop in chain[:-1]:
                notcrop.meta['dsize'] = new_chain_dsize
            new_head = chain[0]
            if TRACE_OPTIMIZE:
                new_head._opt_logs.append('_opt_absorb_overview:absorb_chain')

        warp_meta = ub.dict_isect(self.meta, self._algo_keys)
        tf2 = self.meta['transform']
        dsize = self.meta['dsize']
        new_transform = tf2 @ tf1
        new = self.__class__(new_head, new_transform, dsize=dsize, **warp_meta)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('_opt_absorb_overview:finish')

        if DEBUG:
            print('NEW:')
            new.print_graph()
            print('---------')
        return new

    def _opt_split_warp_overview(self):
        """
        Split this node into a warp and an overview if possible

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image import DelayedLoad
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> self = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> print(f'self={self}')
            >>> print('self.meta = {}'.format(ub.urepr(self.meta, nl=1)))
            >>> warp0 = self.warp({'scale': 0.2})
            >>> warp1 = warp0._opt_split_warp_overview()
            >>> warp2 = self.warp({'scale': 0.25})._opt_split_warp_overview()
            >>> print(ub.urepr(warp0.nesting(), nl=-1, sort=0))
            >>> print(ub.urepr(warp1.nesting(), nl=-1, sort=0))
            >>> print(ub.urepr(warp2.nesting(), nl=-1, sort=0))
            >>> warp0_nodes = [d['type'] for d in warp0.as_graph().nodes.values()]
            >>> warp1_nodes = [d['type'] for d in warp1.as_graph().nodes.values()]
            >>> warp2_nodes = [d['type'] for d in warp2.as_graph().nodes.values()]
            >>> assert warp0_nodes == ['DelayedWarp', 'DelayedLoad']
            >>> assert warp1_nodes == ['DelayedWarp', 'DelayedOverview', 'DelayedLoad']
            >>> assert warp2_nodes == ['DelayedOverview', 'DelayedLoad']

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image import DelayedLoad
            >>> import kwimage
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> self = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> warp0 = self.warp({'scale': 1 / 2 ** 6})
            >>> opt = warp0.optimize()
            >>> print(ub.urepr(warp0.nesting(), nl=-1, sort=0))
            >>> print(ub.urepr(opt.nesting(), nl=-1, sort=0))
            >>> warp0_nodes = [d['type'] for d in warp0.as_graph().nodes.values()]
            >>> opt_nodes = [d['type'] for d in opt.as_graph().nodes.values()]
            >>> assert warp0_nodes == ['DelayedWarp', 'DelayedLoad']
            >>> assert opt_nodes == ['DelayedWarp', 'DelayedOverview', 'DelayedLoad']
        """
        inner_data = self.subdata
        num_overviews = inner_data.num_overviews
        if not num_overviews:
            return self

        # Check if there is a strict downsampling component
        transform = self.meta['transform']
        params = transform.decompose()
        sx, sy = params['scale']
        if sx > 0.5 or sy > 0.5:
            return self

        # Check how many pyramid downs we could replace downsampling with
        num_downs_possible, _, _ = _prepare_scale_residual(sx, sy, fudge=0)
        # But only use as many downs as we have overviews
        num_downs = min(num_overviews, num_downs_possible)
        if num_downs == 0:
            if TRACE_OPTIMIZE:
                self._opt_logs.append('cannot _opt_split_warp_overview')
            return self

        # Given the overview, find the residual to reconstruct the original
        overview_transform = kwimage.Affine.scale(1 / (2 ** num_downs))
        # Let T=original, O=overview, R=residual
        # T = R @ O
        # T @ O.inv = R @ O @ O.inv
        # T @ O.inv = R
        residual_transform = transform @ overview_transform.inv()
        new_transform = residual_transform
        dsize = self.meta['dsize']
        overview = inner_data.get_overview(num_downs)
        if new_transform.isclose_identity():
            new = overview
            if new.dsize != dsize:
                # The warp must have had an implicit crop
                implicit_crop = (slice(0, dsize[1]), slice(0, dsize[0]))
                new = new.crop(implicit_crop, clip=False, wrap=False)
                if TRACE_OPTIMIZE:
                    new._opt_logs.append('Rewrite power of 2 warp as overview with crop')
                new = new.optimize()
            else:
                if TRACE_OPTIMIZE:
                    new._opt_logs.append('Rewrite power of 2 warp as overview')
        else:
            common_meta = ub.dict_isect(self.meta, self._algo_keys)
            new = overview.warp(new_transform, dsize=dsize, **common_meta)
            if TRACE_OPTIMIZE:
                new._opt_logs.append('Factor out overview from warp')
        if TRACE_OPTIMIZE:
            new._opt_logs.append('_opt_split_warp_overview')
        return new


def _prepare_scale_residual(sx, sy, fudge=0):
    # Previously in:
    # from kwimage.im_cv2 import _prepare_scale_residual
    # May be moved to a common kwimage utility module
    max_scale = max(sx, sy)
    ideal_num_downs = int(np.log2(1 / max_scale))
    num_downs = max(ideal_num_downs - fudge, 0)
    pyr_scale = 1 / (2 ** num_downs)
    residual_sx = sx / pyr_scale
    residual_sy = sy / pyr_scale
    return num_downs, residual_sx, residual_sy


class DelayedDequantize(DelayedImage):
    """
    Rescales image intensities from int to floats.

    The output is usually between 0 and 1. This also handles transforming
    nodata into nan values.
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedImage.__slots__
    def __init__(self, subdata, quantization):
        """
        Args:
            subdata (DelayedArray): data to operate on
            quantization (Dict):
                see :func:`delayed_image.helpers.dequantize`
        """
        super().__init__(subdata)
        self.meta['quantization'] = quantization
        self.meta['dsize'] = subdata.dsize

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        from delayed_image.helpers import dequantize
        quantization = self.meta['quantization']
        final = self.subdata._finalize()
        final = kwarray.atleast_nd(final, 3, front=False)
        if quantization is not None:
            final = dequantize(final, quantization)
        return final

    def optimize(self):
        """

        Returns:
            DelayedImage

        Example:
            >>> # Test a case that caused an error in development
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image import DelayedLoad
            >>> fpath = kwimage.grab_test_image_fpath()
            >>> base = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> quantization = {'quant_max': 255, 'nodata': 0}
            >>> self = base.get_overview(1).dequantize(quantization)
            >>> self.write_network_text()
            >>> opt = self.optimize()
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()

        if isinstance2(new.subdata, DelayedDequantize):
            raise AssertionError('Dequantization is only allowed once')

        if isinstance2(new.subdata, DelayedWarp):
            # Swap order so quantize is before the warp
            new = new._opt_dequant_before_other()
            new = new.optimize()

        if isinstance2(new.subdata, DelayedChannelConcat):
            new = new._opt_push_under_concat().optimize()
        if TRACE_OPTIMIZE:
            new._opt_logs.append('optimize DelayedDequantize')
        return new

    def _opt_dequant_before_other(self):
        quantization = self.meta['quantization']
        new = copy.copy(self.subdata)
        new.subdata = new.subdata.dequantize(quantization)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('_opt_dequant_before_other')
        return new

    def _transform_from_subdata(self):
        return kwimage.Affine.eye()


class DelayedCrop(DelayedImage):
    """
    Crops an image along integer pixel coordinates.

    Example:
        >>> from delayed_image.delayed_nodes import *  # NOQA
        >>> from delayed_image import DelayedLoad
        >>> base = DelayedLoad.demo(dsize=(16, 16)).prepare()
        >>> # Test Fuse Crops Space Only
        >>> crop1 = base[4:12, 0:16]
        >>> self = crop1[2:6, 0:8]
        >>> opt = self._opt_fuse_crops()
        >>> self.write_network_text()
        >>> opt.write_network_text()
        >>> #
        >>> # Test Channel Select Via Index
        >>> self = base[:, :, [0]]
        >>> self.write_network_text()
        >>> final = self._finalize()
        >>> assert final.shape == (16, 16, 1)
        >>> assert base[:, :, [0, 1]].finalize().shape == (16, 16, 2)
        >>> assert base[:, :, [2, 0, 1]].finalize().shape == (16, 16, 3)

    Example:
        >>> from delayed_image.delayed_nodes import *  # NOQA
        >>> from delayed_image import DelayedLoad
        >>> base = DelayedLoad.demo(dsize=(16, 16)).prepare()
        >>> # Test Discontiguous Channel Select Via Index
        >>> self = base[:, :, [0, 2]]
        >>> self.write_network_text()
        >>> final = self._finalize()
        >>> assert final.shape == (16, 16, 2)
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedImage.__slots__

    def __init__(self, subdata, space_slice=None, chan_idxs=None):
        """
        Args:
            subdata (DelayedArray): data to operate on

            space_slice (Tuple[slice, slice]):
                if speficied, take this y-slice and x-slice.

            chan_idxs (List[int] | None):
                if specified, take these channels / bands
        """
        super().__init__(subdata)
        # TODO: are we doing infinite padding or clipping?
        # This assumes clipping
        in_w, in_h = subdata.dsize
        if space_slice is not None:
            space_dims = (in_h, in_w)
            slice_box = kwimage.Boxes.from_slice(
                space_slice, space_dims, wrap=True, clip=True)
            space_slice = slice_box.to_slices()[0]
            space_slice, _pad = kwarray.embed_slice(space_slice, space_dims)
            sl_y, sl_x = space_slice[0:2]
            width = sl_x.stop - sl_x.start
            height = sl_y.stop - sl_y.start
            self.meta['dsize'] = (width, height)
        else:
            space_slice = (slice(0, in_h), slice(0, in_w))
            self.meta['dsize'] = (in_w, in_h)

        if chan_idxs is not None:
            current_channels = self.channels
            if current_channels is not None:
                new_channels = current_channels[chan_idxs]
            else:
                new_channels = len(chan_idxs)
            self.channels = new_channels

        self.meta['space_slice'] = space_slice
        self.meta['chan_idxs'] = chan_idxs

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        space_slice = self.meta['space_slice']
        chan_idxs = self.meta['chan_idxs']
        sub_final = self.subdata._finalize()
        if chan_idxs is None:
            full_slice = space_slice
        else:
            full_slice = space_slice + (chan_idxs,)
        # final = sub_final[space_slice]
        final = sub_final[full_slice]
        final = kwarray.atleast_nd(final, 3)
        return final

    def _transform_from_subdata(self):
        sl_y, sl_x = self.meta['space_slice']
        offset = -sl_x.start, -sl_y.start
        self_from_subdata = kwimage.Affine.translate(offset)
        return self_from_subdata

    def optimize(self):
        """
        Returns:
            DelayedImage

        Example:
            >>> # Test optimize nans
            >>> from delayed_image import DelayedNans
            >>> import kwimage
            >>> base = DelayedNans(dsize=(100, 100), channels='a|b|c')
            >>> self = base[0:10, 0:5]
            >>> # Should simply return a new nan generator
            >>> new = self.optimize()
            >>> self.write_network_text()
            >>> new.write_network_text()
            >>> assert len(new.as_graph().nodes) == 1
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance2(new.subdata, DelayedCrop):
            new = new._opt_fuse_crops()

        if hasattr(new.subdata, '_optimized_crop'):
            # The subdata knows how to optimize itself wrt this node
            crop_kwargs = ub.dict_isect(self.meta, {'space_slice', 'chan_idxs'})
            new = new.subdata._optimized_crop(**crop_kwargs).optimize()
        if isinstance2(new.subdata, DelayedWarp):
            new = new._opt_warp_after_crop()
            new = new.optimize()
        elif isinstance2(new.subdata, DelayedDequantize):
            new = new._opt_dequant_after_crop()
            new = new.optimize()

        if isinstance2(new.subdata, DelayedChannelConcat):
            if isinstance2(new, DelayedCrop):
                # We have to be careful if there we have band selection
                chan_idxs = new.meta.get('chan_idxs', None)
                space_slice = new.meta.get('space_slice', None)
                taken = new.subdata
                if TRACE_OPTIMIZE:
                    _new_logs = []
                if chan_idxs is not None:
                    if TRACE_OPTIMIZE:
                        _new_logs.extend(new.subdata._opt_logs)
                        _new_logs.extend(new._opt_logs)
                        _new_logs.append('concat-chan-crop-interact')
                    taken = new.subdata.take_channels(chan_idxs).optimize()
                if space_slice is not None:
                    if TRACE_OPTIMIZE:
                        _new_logs.append('concat-space-crop-interact')
                    taken = taken.crop(space_slice)._opt_push_under_concat().optimize()
                new = taken
                if TRACE_OPTIMIZE:
                    new._opt_logs.extend(_new_logs)
            else:
                new = new._opt_push_under_concat().optimize()
        if TRACE_OPTIMIZE:
            new._opt_logs.append('optimize crop')
        return new

    def _opt_fuse_crops(self):
        """
        Combine two consecutive crops into a single operation.

        Example:
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> base = DelayedLoad.demo(dsize=(16, 16)).prepare()
            >>> # Test Fuse Crops Space Only
            >>> crop1 = base[4:12, 0:16]
            >>> crop2 = self = crop1[2:6, 0:8]
            >>> opt = crop2._opt_fuse_crops()
            >>> self.write_network_text()
            >>> opt.write_network_text()
            >>> opt._validate()
            >>> self._validate()

        Example:
            >>> # Test Fuse Crops Channels Only
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> base = DelayedLoad.demo(dsize=(16, 16)).prepare()
            >>> crop1 = base.crop(chan_idxs=[0, 2, 1])
            >>> crop2 = crop1.crop(chan_idxs=[1, 2])
            >>> crop3 = self = crop2.crop(chan_idxs=[0, 1])
            >>> opt = self._opt_fuse_crops()._opt_fuse_crops()
            >>> self.write_network_text()
            >>> opt.write_network_text()
            >>> finalB = base._validate()._finalize()
            >>> final1 = opt._validate()._finalize()
            >>> final2 = self._validate()._finalize()
            >>> assert np.all(final2[..., 0] == finalB[..., 2])
            >>> assert np.all(final2[..., 1] == finalB[..., 1])
            >>> assert np.all(final2[..., 0] == final1[..., 0])
            >>> assert np.all(final2[..., 1] == final1[..., 1])

        Example:
            >>> # Test Fuse Crops Space  And Channels
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> base = DelayedLoad.demo(dsize=(16, 16)).prepare()
            >>> crop1 = base[4:12, 0:16, [1, 2]]
            >>> self = crop1[2:6, 0:8, [1]]
            >>> opt = self._opt_fuse_crops()
            >>> self.write_network_text()
            >>> opt.write_network_text()
            >>> self._validate()
            >>> crop1._validate()
        """
        assert isinstance2(self.subdata, DelayedCrop), (
            'can only call this method on two sequential crops'
        )
        # self.print_graph()
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        inner_data = self.subdata.subdata

        inner_slices = self.subdata.meta['space_slice']
        outer_slices = self.meta['space_slice']

        outer_ysl, outer_xsl = outer_slices
        inner_ysl, inner_xsl = inner_slices

        # Apply the new relative slice to the current absolute slice
        new_xstart = min(inner_xsl.start + outer_xsl.start, inner_xsl.stop)
        new_xstop = min(inner_xsl.start + outer_xsl.stop, inner_xsl.stop)
        new_ystart = min(inner_ysl.start + outer_ysl.start, inner_ysl.stop)
        new_ystop = min(inner_ysl.start + outer_ysl.stop, inner_ysl.stop)

        # Handle bands
        inner_chan_idxs = self.subdata.meta['chan_idxs']
        outer_chan_idxs = self.meta['chan_idxs']
        if outer_chan_idxs is None and inner_chan_idxs is None:
            new_chan_idxs = None
        elif outer_chan_idxs is None:
            new_chan_idxs = inner_chan_idxs
        elif inner_chan_idxs is None:
            new_chan_idxs = outer_chan_idxs
        else:
            new_chan_idxs = list(ub.take(inner_chan_idxs, outer_chan_idxs))
        new_crop = (slice(new_ystart, new_ystop), slice(new_xstart, new_xstop))
        new = self.__class__(inner_data, new_crop, new_chan_idxs)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('Fuse crops')
        return new

    def _opt_warp_after_crop(self):
        """
        If the child node is a warp, move it after the crop.

        This is more efficient because:
            1. The crop is closer to the load.
            2. we are warping with less data.

        Example:
            >>> from delayed_image.delayed_nodes import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> fpath = kwimage.grab_test_image_fpath()
            >>> node0 = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> node1 = node0.warp({'scale': 0.432,
            >>>                     'theta': np.pi / 3,
            >>>                     'about': (80, 80),
            >>>                     'shearx': .3,
            >>>                     'offset': (-50, -50)})
            >>> node2 = node1[10:50, 1:40]
            >>> self = node2
            >>> new_outer = node2._opt_warp_after_crop()
            >>> print(ub.urepr(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.urepr(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self._finalize()
            >>> final1 = new_outer._finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(2, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(2, 2, 2), fnum=1, title='optimized')

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from delayed_image import *  # NOQA
            >>> from delayed_image.delayed_leafs import DelayedLoad
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> node1 = node0.warp({'scale': 1000 / 512})
            >>> node2 = node1[250:750, 0:500]
            >>> self = node2
            >>> new_outer = node2._opt_warp_after_crop()
            >>> print(ub.urepr(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.urepr(new_outer.nesting(), nl=-1, sort=0))
        """
        assert isinstance2(self.subdata, DelayedWarp)
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        outer_slices = self.meta['space_slice']
        outer_chan_idxs = self.meta['chan_idxs']
        inner_transform = self.subdata.meta['transform']

        outer_region = kwimage.Boxes.from_slice(outer_slices)
        outer_region = outer_region.to_polygons()[0]

        from delayed_image.helpers import _swap_warp_after_crop
        # Should origin_convention be configurable? I think no for now.
        inner_slice, outer_transform = _swap_warp_after_crop(
            outer_region, inner_transform, origin_convention='corner')

        warp_meta = ub.dict_isect(self.meta, {'dsize'})
        warp_meta.update(ub.dict_isect(
            self.subdata.meta, self.subdata._algo_keys))

        new_inner = self.subdata.subdata.crop(inner_slice, outer_chan_idxs)
        new_outer = new_inner.warp(outer_transform, **warp_meta)
        if TRACE_OPTIMIZE:
            new_outer._opt_logs.extend(self.subdata.subdata._opt_logs)
            new_outer._opt_logs.extend(self.subdata._opt_logs)
            new_outer._opt_logs.append('_opt_warp_after_crop')
        return new_outer

    def _opt_dequant_after_crop(self):
        # Swap order so dequantize is after the crop
        assert isinstance2(self.subdata, DelayedDequantize)
        quantization = self.subdata.meta['quantization']
        new = copy.copy(self)
        new.subdata = self.subdata.subdata  # Remove the dequantization
        new = new.dequantize(quantization)  # Push it after the crop
        if TRACE_OPTIMIZE:
            new._opt_logs.append('_opt_dequant_after_crop')
        return new


class DelayedOverview(DelayedImage):
    """
    Downsamples an image by a factor of two.

    If the underlying image being loaded has precomputed overviews it simply
    loads these instead of downsampling the original image, which is more
    efficient.

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Make a complex chain of operations and optimize it
        >>> from delayed_image import *  # NOQA
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
        >>> dimg = DelayedLoad(fpath, channels='r|g|b').prepare()
        >>> dimg = dimg.get_overview(1)
        >>> dimg = dimg.get_overview(1)
        >>> dimg = dimg.get_overview(1)
        >>> dopt = dimg.optimize()
        >>> if 1:
        >>>     import networkx as nx
        >>>     dimg.write_network_text()
        >>>     dopt.write_network_text()
        >>> print(ub.urepr(dopt.nesting(), nl=-1, sort=0))
        >>> final0 = dimg._finalize()[:]
        >>> final1 = dopt._finalize()[:]
        >>> assert final0.shape == final1.shape
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
        >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedImage.__slots__

    def __init__(self, subdata, overview):
        """
        Args:
            subdata (DelayedArray): data to operate on
            overview (int): the overview to use (assuming it exists)
        """
        super().__init__(subdata)
        self.meta['overview'] = overview
        w, h = subdata.dsize
        sf = 1 / (2 ** overview)
        """
        Ignore:
            # Check how gdal handles overviews for odd sized images.
            imdata = np.random.rand(31, 29)
            kwimage.imwrite('foo.tif', imdata, backend='gdal', overviews=3)
            ub.cmd('gdalinfo foo.tif', verbose=3)
        """
        # The rounding operation for gdal overviews is ceiling
        w = iceil(sf * w)
        h = iceil(sf * h)
        self.meta['dsize'] = (w, h)

    @property
    def num_overviews(self):
        """
        Returns:
            int
        """
        # This operation reduces the number of available overviews
        num_remain = self.subdata.num_overviews - self.meta['overview']
        return num_remain

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        sub_final = self.subdata._finalize()
        overview = self.meta['overview']
        if hasattr(sub_final, 'get_overview'):
            # This should be a lazy gdal frame
            if sub_final.num_overviews >= overview:
                final = sub_final.get_overview(overview)
                return final

        warnings.warn(ub.paragraph(
            '''
            The underlying data does not have overviews.
            Simulating the overview using a imresize operation.
            '''
        ))
        sub_final = np.asarray(sub_final)
        final = kwimage.imresize(
            sub_final,
            scale=1 / 2 ** overview,
            interpolation='nearest',
            # antialias=True
        )
        return final

    def optimize(self):
        """
        Returns:
            DelayedImage
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance2(new.subdata, DelayedOverview):
            new = new._opt_fuse_overview()

        if new.meta['overview'] == 0:
            new = new.subdata
        elif isinstance2(new.subdata, DelayedCrop):
            new = new._opt_crop_after_overview()
            new = new.optimize()
        elif isinstance2(new.subdata, DelayedWarp):
            new = new._opt_warp_after_overview()
            new = new.optimize()
        elif isinstance2(new.subdata, DelayedDequantize):
            new = new._opt_dequant_after_overview()
            new = new.optimize()
        if isinstance2(new.subdata, DelayedChannelConcat):
            new = new._opt_push_under_concat().optimize()
        if TRACE_OPTIMIZE:
            new._opt_logs.append('optimize overview')
        return new

    def _transform_from_subdata(self):
        scale = 1 / 2 ** self.meta['overview']
        return kwimage.Affine.scale(scale)

    def _opt_overview_as_warp(self):
        """
        Sometimes it is beneficial to replace an overview with a warp as an
        intermediate optimization step.
        """
        transform = self._transform_from_subdata()
        dsize = self.meta['dsize']
        new = self.subdata.warp(transform, dsize=dsize)
        if TRACE_OPTIMIZE:
            new._opt_logs.append(self._opt_logs)
            new._opt_logs.append('_opt_overview_as_warp')
        return new

    def _opt_crop_after_overview(self):
        """
        Given an outer overview and an inner crop, switch places. We want the
        overview to be as close to the load as possible.

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from delayed_image import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> node1 = node0[100:400, 120:450]
            >>> node2 = node1.get_overview(2)
            >>> self = node2
            >>> new_outer = node2.optimize()
            >>> print(ub.urepr(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.urepr(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self._finalize()
            >>> final1 = new_outer._finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from delayed_image import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> node1 = node0[:, :, 0:2]
            >>> node2 = node1.get_overview(2)
            >>> self = node2
            >>> new_outer = node2.optimize()
            >>> node2.write_network_text()
            >>> new_outer.write_network_text()
            >>> assert node2.shape[2] == 2
            >>> assert new_outer.shape[2] == 2
        """
        from delayed_image.helpers import _swap_crop_after_warp
        assert isinstance2(self.subdata, DelayedCrop)
        # Inner is the data closer to the leaf (disk), outer is the data closer
        # to the user (output).
        outer_overview = self.meta['overview']
        inner_slices = self.subdata.meta['space_slice']
        chan_idxs = self.subdata.meta['chan_idxs']

        sf = 1 / 2 ** outer_overview
        outer_transform = kwimage.Affine.scale(sf)

        inner_region = kwimage.Boxes.from_slice(inner_slices)
        inner_region = inner_region.to_polygons()[0]

        new_inner_warp, outer_crop, new_outer_warp = _swap_crop_after_warp(
            inner_region, outer_transform)

        # Move the overview to the inside, it should be unchanged
        new = self.subdata.subdata.get_overview(outer_overview)

        # Move the crop to the outside
        new = new.crop(outer_crop, chan_idxs=chan_idxs)

        if not np.all(np.isclose(np.eye(3), new_outer_warp)):
            # we might have to apply an additional warp at the end.
            new = new.warp(new_outer_warp)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('_opt_crop_after_overview')
        return new

    def _opt_fuse_overview(self):
        assert isinstance2(self.subdata, DelayedOverview)
        outer_overview = self.meta['overview']
        inner_overrview = self.subdata.meta['overview']
        new_overview = inner_overrview + outer_overview
        new = self.subdata.subdata.get_overview(new_overview)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('_opt_fuse_overview')
        return new

    def _opt_dequant_after_overview(self):
        # Swap order so dequantize is after the crop
        assert isinstance2(self.subdata, DelayedDequantize)
        quantization = self.subdata.meta['quantization']
        new = copy.copy(self)
        new.subdata = self.subdata.subdata  # Remove the dequantization
        new = new.dequantize(quantization)  # Push it after the crop
        if TRACE_OPTIMIZE:
            new._opt_logs.append('_opt_dequant_after_overview')
        return new

    def _opt_warp_after_overview(self):
        """
        Given an warp followed by an overview, move the warp to the outer scope
        such that the overview is first.

        Example:
            >>> # xdoctest: +REQUIRES(module:osgeo)
            >>> from delayed_image import *  # NOQA
            >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
            >>> node0 = DelayedLoad(fpath, channels='r|g|b').prepare()
            >>> node1 = node0.warp({'scale': (2.1, .7), 'offset': (20, 40)})
            >>> node2 = node1.get_overview(2)
            >>> self = node2
            >>> new_outer = node2.optimize()
            >>> print(ub.urepr(node2.nesting(), nl=-1, sort=0))
            >>> print(ub.urepr(new_outer.nesting(), nl=-1, sort=0))
            >>> final0 = self._finalize()
            >>> final1 = new_outer._finalize()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
            >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')
        """
        assert isinstance2(self.subdata, DelayedWarp)
        outer_overview = self.meta['overview']
        inner_transform = self.subdata.meta['transform']
        outer_transform = self._transform_from_subdata()
        A = outer_transform
        B = inner_transform
        # We have: A @ B, and we want that to equal C @ A
        # where the overview A left as-is and moved inside, we modify the new
        # outer transform C to accomodate this.
        # So C = A @ B @ A.inv()
        C = A @ B @ A.inv()
        new_outer = C
        new_inner_overview = outer_overview
        new = self.subdata.subdata.get_overview(new_inner_overview)
        new = new.warp(new_outer)
        if TRACE_OPTIMIZE:
            new._opt_logs.extend(self.subdata._opt_logs)
            new._opt_logs.append('_opt_warp_after_overview')
        return new


def _rectify_retain(remove, retain):
    if remove is not None or retain is None:
        valid_keys = {"offset", "scale", "shearx", "theta"}
        if remove is not None and retain is not None:
            raise ValueError('retain and remove are mutually exclusive')
        if remove is not None:
            retain = valid_keys - set(remove)
        else:
            if retain is None:
                retain = set()
    return retain


class CoordinateCompatibilityError(ValueError):
    """
    Error when trying to perform operations on data in different coordinate
    systems.
    """


class _InnerAccumSegment:
    """
    Gather the indexes we need to take from an inner component

    This is a helper for :func:`DelayedChannelConcat.take_channels`
    """
    if delayed_base.USE_SLOTS:
        __slots__ = ('comp', 'start', 'stop', 'codes', 'indexes')

    def __init__(curr, comp):
        curr.comp = comp
        curr.start = None
        curr.stop = None
        curr.codes = None
        curr.indexes = []

    def add_inner(curr, inner, code):
        if curr.start is None:
            curr.start = inner
            curr.stop = inner + 1
            if code is not None:
                curr.codes = []
        else:
            if code is None:
                curr.codes = None
            # Can we take a contiguous slice?
            if curr.stop is not None and curr.stop == inner:
                curr.stop = inner + 1
            else:
                # Contiguous input is broken, fallback to indexes only.
                curr.stop = None

        # Accumulate the codes if we can
        if curr.codes is not None:
            curr.codes.append(code)

        # cast to int to prevent numpy types
        curr.indexes.append(int(inner))

    def get_indexer(curr):
        """
        Return an list of indexes into the subcomponent that will form
        the new contiguous out-component
        """
        if curr.stop is not None:
            if curr.start == 0 and curr.stop == curr.comp.num_channels:
                # We can take the entire component
                sub_idxs = Ellipsis
                assert curr.indexes[0] == 0
                assert curr.indexes[-1] == curr.stop - 1
            else:
                # In this case we could take the parts as a contiguous
                # slice, but just return indexes for now
                sub_idxs = list(range(curr.start, curr.stop))
                assert sub_idxs == curr.indexes
        else:
            sub_idxs = curr.indexes
        return sub_idxs

    def get_subcomponent(curr, dsize):
        """
        Finalize the subcomponent
        """
        from delayed_image.delayed_leafs import DelayedNans
        comp = curr.comp
        if comp is None:
            # There is no component to get a subcomponent from.
            # Instead, return nans that correspond to the requested
            # codes.
            if curr.codes is None:
                nan_chan = None
            else:
                nan_chan = FusedChannelSpec(curr.codes)
            sub_comp = DelayedNans(dsize, channels=nan_chan)
        else:
            sub_idxs = curr.get_indexer()
            if sub_idxs is Ellipsis:
                # Entire component is valid, no need for sub-operation
                sub_comp = comp
            else:
                sub_comp = comp.take_channels(sub_idxs)
        return sub_comp


if IS_DEVELOPING:
    def isinstance2(inst, cls):
        """
        In production regular isinstance works fine, but when debugging in IPython
        reloading classes will causes it to break, so we special case it here.

        Args:
            item (object): instance to check
            cls (type): class to check against

        Returns:
            bool

        Ignore:
            from delayed_image.delayed_nodes import *  # NOQA
            from delayed_image.delayed_leafs import DelayedNans

            inst = DelayedNans()
            cls = DelayedImage
            isinstance2(inst, cls)
            isinstance2(inst, DelayedWarp)
        """
        import sys
        USE_REAL_ISINSTANCE = 'IPython' not in sys.modules
        if USE_REAL_ISINSTANCE:
            return isinstance(inst, cls)
        else:
            return any(inst_cls.__name__ == cls.__name__
                       for inst_cls in inst.__class__.__mro__)
else:
    isinstance2 = isinstance


def iceil(x):
    return int(np.ceil(x))

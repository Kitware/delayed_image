"""
Terminal nodes
"""

import kwarray
import kwimage
import numpy as np
import warnings
from delayed_image import delayed_nodes
from delayed_image import delayed_base

__docstubs__ = """
from delayed_image.channel_spec import FusedChannelSpec
"""


TRACE_OPTIMIZE = delayed_nodes.TRACE_OPTIMIZE


class DelayedImageLeaf(delayed_nodes.DelayedImage):
    if delayed_base.USE_SLOTS:
        __slots__ = delayed_nodes.DelayedImage.__slots__

    def get_transform_from_leaf(self):
        """
        Returns the transformation that would align data with the leaf

        Returns:
            kwimage.Affine
        """
        return kwimage.Affine.eye()

    def optimize(self):
        if TRACE_OPTIMIZE:
            self._opt_logs.append('optimize DelayedImageLeaf')
        return self


class DelayedLoad(DelayedImageLeaf):
    """
    Points to an image on disk to be loaded.

    This is the starting point for most delayed operations. Disk IO is avoided
    until the ``finalize`` operation is called. Calling ``prepare`` can read
    image headers if metadata like the image width, height, and number of
    channels is not provided, but most operations can be performed while these
    are still unknown.

    If a gdal backend is available, and the underlying image is in the
    appropriate formate (e.g. COG), finalize will return a lazy reference that
    enables fast overviews and crops. For image formats that do not allow for
    tiling / overviews, then there is no way to avoid reading entire image as
    an ndarray.

    Example:
        >>> from delayed_image import *  # NOQA
        >>> self = DelayedLoad.demo(dsize=(16, 16)).prepare()
        >>> data1 = self.finalize()

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Demo code to develop support for overviews
        >>> from delayed_image import *  # NOQA
        >>> import kwimage
        >>> import ubelt as ub
        >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
        >>> self = DelayedLoad(fpath, channels='r|g|b').prepare()
        >>> print(f'self={self}')
        >>> print('self.meta = {}'.format(ub.repr2(self.meta, nl=1)))
        >>> quantization = {
        >>>     'quant_max': 255,
        >>>     'nodata': 0,
        >>> }
        >>> node0 = self
        >>> node1 = node0.get_overview(2)
        >>> node2 = node1[13:900, 11:700]
        >>> node3 = node2.dequantize(quantization)
        >>> node4 = node3.warp({'scale': 0.05})
        >>> #
        >>> data0 = node0._validate().finalize()
        >>> data1 = node1._validate().finalize()
        >>> data2 = node2._validate().finalize()
        >>> data3 = node3._validate().finalize()
        >>> data4 = node4._validate().finalize()
        >>> node4.write_network_text()

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Test delayed ops with int16 and nodata values
        >>> from delayed_image import *  # NOQA
        >>> import kwimage
        >>> from delayed_image.helpers import quantize_float01
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('delayed_image/tests/test_delay_nodata').ensuredir()
        >>> fpath = dpath / 'data.tif'
        >>> data = kwimage.ensure_float01(kwimage.grab_test_image())
        >>> poly = kwimage.Polygon.random(rng=321032).scale(data.shape[0])
        >>> poly.fill(data, np.nan)
        >>> data_uint16, quantization = quantize_float01(data)
        >>> nodata = quantization['nodata']
        >>> kwimage.imwrite(fpath, data_uint16, nodata_value=nodata, backend='gdal', overviews=3)
        >>> # Test loading the data
        >>> self = DelayedLoad(fpath, channels='r|g|b', nodata_method='float').prepare()
        >>> node0 = self
        >>> node1 = node0.dequantize(quantization)
        >>> node2 = node1.warp({'scale': 0.51}, interpolation='lanczos')
        >>> node3 = node2[13:900, 11:700]
        >>> node4 = node3.warp({'scale': 0.9}, interpolation='lanczos')
        >>> node4.write_network_text()
        >>> node5 = node4.optimize()
        >>> node5.write_network_text()
        >>> node6 = node5.warp({'scale': 8}, interpolation='lanczos').optimize()
        >>> node6.write_network_text()
        >>> #
        >>> data0 = node0._validate().finalize()
        >>> data1 = node1._validate().finalize()
        >>> data2 = node2._validate().finalize()
        >>> data3 = node3._validate().finalize()
        >>> data4 = node4._validate().finalize()
        >>> data5 = node5._validate().finalize()
        >>> data6 = node6._validate().finalize()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> stack1 = kwimage.stack_images([data1, data2, data3, data4, data5])
        >>> stack2 = kwimage.stack_images([stack1, data6], axis=1)
        >>> kwplot.imshow(stack2)
    """
    if delayed_base.USE_SLOTS:
        __slots__ = delayed_nodes.DelayedImage.__slots__ + ('lazy_ref',)

    def __init__(self, fpath, channels=None, dsize=None, nodata_method=None, num_overviews=None):
        """
        Args:
            fpath (str | PathLike):
                URI pointing at the image data to load

            channels (int | str | FusedChannelSpec | None):
                the underlying channels of the image if known a-priori

            dsize (Tuple[int, int]):
                The width / height of the image if known a-priori

            nodata_method (str | None):
                How to handle nodata values in the file itself.
                Can be "auto", "float", or "ma".

            num_overviews (int | None):
                number of overviews if known a-priori
        """
        super().__init__(channels=channels, dsize=dsize)
        self.meta['fpath'] = fpath
        self.meta['nodata_method'] = nodata_method
        self.meta['num_overviews'] = num_overviews
        self.lazy_ref = None

    @property
    def fpath(self):
        return self.meta['fpath']

    @classmethod
    def demo(DelayedLoad, key='astro', channels=None, dsize=None,
             nodata_method=None, overviews=None):
        """
        Creates a demo DelayedLoad node that points to a file generated by
        :func:`kwimage.grab_test_image_fpath`.

        If metadata like dsize and channels are not provided, then the
        :func:`prepare` can be used to auto-populate them at the cost of the
        disk IO to read image headers.

        Args:
            key (str): which test image to grab. Valid choices are:
                astro - an astronaught
                carl - Carl Sagan
                paraview - ParaView logo
                stars - picture of stars in the sky

            channels (str):
                if specified, these channels will be stored in the delayed load
                metadata. Note: these are not auto-populated. Usually the
                key corresponds to 3-channel data,

            dsize (None | Tuple[int, int]):
                if specified, we will return a variant of the data with the
                specific dsize

            nodata_method (str | None):
                How to handle nodata values in the file itself.
                Can be "auto", "float", or "ma".

            overviews (None | int):
                if specified, will return a variant of the data with overviews

        Returns:
            DelayedLoad

        Example:
            >>> from delayed_image.delayed_leafs import *  # NOQA
            >>> import delayed_image
            >>> delayed = delayed_image.DelayedLoad.demo()
            >>> print(f'delayed={delayed}')
            >>> delayed.prepare()
            >>> print(f'delayed={delayed}')
            >>> delayed = DelayedLoad.demo(channels='r|g|b', nodata_method='float')
            >>> print(f'delayed={delayed}')
            >>> delayed.prepare()
            >>> print(f'delayed={delayed}')
            >>> delayed.finalize()
        """
        fpath = kwimage.grab_test_image_fpath(key, dsize=dsize,
                                              overviews=overviews)
        self = DelayedLoad(fpath, channels=channels, dsize=dsize,
                           nodata_method=nodata_method)
        return self

    def _load_reference(self):
        if self.lazy_ref is None:
            from delayed_image import lazy_loaders
            using_gdal = lazy_loaders.LazyGDalFrameFile.available()
            nodata_method = self.meta.get('nodata_method', None)
            if using_gdal:
                self.lazy_ref = lazy_loaders.LazyGDalFrameFile(
                    self.fpath, nodata_method=nodata_method)
            else:
                if nodata_method == 'auto':
                    raise Exception('need gdal for auto no-data')
                self.lazy_ref = NotImplemented
        return self

    def prepare(self):
        """
        If metadata is missing, perform minimal IO operations in order to
        prepopulate metadata that could help us better optimize the operation
        tree.

        Returns:
            DelayedLoad
        """
        self._load_metadata()
        return self

    def _load_metadata(self):
        """
        Ignore:
            We want to be able to skip the reference load if the metadata is
            already setup.

            import kwimage
            from delayed_image.delayed_leafs import *  # NOQA
            dsize = (32, 32)
            channels = 'red|green|blue'
            fpath = kwimage.grab_test_image_fpath('astro', dsize=dsize)
            self = DelayedLoad(fpath, channels=channels, dsize=dsize, num_overviews=1)
            xdev.profile_now(self._load_metadata)()

        """
        required_meta_keys = ('dsize', 'num_channels', 'num_overviews')
        if all(self.meta[k] is not None for k in required_meta_keys):
            if not any(d is None for d in self.dsize):
                return self
        self._load_reference()
        if self.lazy_ref is NotImplemented:
            shape = kwimage.load_image_shape(self.fpath)
            if len(shape) == 2:
                shape = shape + (1,)
            num_overviews = 0
        else:
            shape = self.lazy_ref.shape
            num_overviews = self.lazy_ref.num_overviews
        h, w, c = shape
        if self.dsize is None or any(d is None for d in self.dsize):
            self.meta['dsize'] = (w, h)
        if self.num_channels is None:
            self.meta['num_channels'] = c
        self.meta['num_overviews'] = num_overviews
        return self

    def _finalize(self):
        """
        Returns:
            ArrayLike

        Example:
            >>> # Check difference between finalize and _finalize
            >>> from delayed_image.delayed_leafs import *  # NOQA
            >>> self = DelayedLoad.demo().prepare()
            >>> final_arr = self.finalize()
            >>> assert isinstance(final_arr, np.ndarray), 'finalize should always return an array'
            >>> final_ref = self._finalize()
            >>> if self.lazy_ref is not NotImplemented:
            >>>     assert not isinstance(final_ref, np.ndarray), (
            >>>         'A pure load with gdal should return a reference that is '
            >>>         'similiar to but not quite an array')
        """
        self._load_reference()
        if self.lazy_ref is NotImplemented:
            warnings.warn('DelayedLoad may not be efficient without gdal')
            pre_final = kwimage.imread(self.fpath)
            pre_final = kwarray.atleast_nd(pre_final, 3)
            return pre_final
        else:
            # Need to ensure that if any metadata changed, we modify the
            # underlying lazy ref.
            self.lazy_ref.nodata_method = self.meta.get('nodata_method', None)
            return self.lazy_ref

    # Reimplementations of existing properties with specialized logic for speed
    # @property
    # def num_channels(self):
    #     """
    #     Returns:
    #         None | int
    #     """
    #     return self.meta.get('num_channels', None)

    # @property
    # def dsize(self):
    #     """
    #     Returns:
    #         None | Tuple[int | None, int | None]
    #     """
    #     return self.meta.get('dsize', None)

    # @property
    # def channels(self):
    #     """
    #     Returns:
    #         None | FusedChannelSpec
    #     """
    #     return self.meta.get('channels', None)


class DelayedNans(DelayedImageLeaf):
    """
    Constructs nan channels as needed

    Example:
        self = DelayedNans((10, 10), channel_spec.FusedChannelSpec.coerce('rgb'))
        region_slices = (slice(5, 10), slice(1, 12))
        delayed = self.crop(region_slices)

    Example:
        >>> from delayed_image.delayed_leafs import *  # NOQA
        >>> from delayed_image import DelayedChannelConcat
        >>> dsize = (307, 311)
        >>> c1 = DelayedNans(dsize=dsize, channels='foo')
        >>> c2 = DelayedLoad.demo('astro', dsize=dsize, channels='R|G|B').prepare()
        >>> cat = DelayedChannelConcat([c1, c2])
        >>> warped_cat = cat.warp({'scale': 1.07}, dsize=(328, 332))._validate()
        >>> warped_cat._validate().optimize().finalize()
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedImageLeaf.__slots__ + ('_kwargs',)

    def __init__(self, dsize=None, channels=None):
        super().__init__(channels=channels, dsize=dsize)
        self._kwargs = {}

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        shape = self.shape
        from delayed_image.helpers import _ensure_valid_shape
        shape = _ensure_valid_shape(shape)
        final = np.full(shape, fill_value=np.nan)
        return final

    def _optimized_crop(self, space_slice=None, chan_idxs=None):
        """
        Crops an image along integer pixel coordinates.

        Args:
            space_slice (Tuple[slice, slice]): y-slice and x-slice.
            chan_idxs (List[int]): indexes of bands to take

        Returns:
            DelayedImage
        """
        if chan_idxs is None:
            channels = self.channels
        else:
            channels = self.channels[chan_idxs]
        dsize = self.dsize
        data_dims = dsize[::-1]
        data_slice, extra_pad = kwarray.embed_slice(space_slice, data_dims)
        box = kwimage.Boxes.from_slice(data_slice)
        new_width = box.width.ravel()[0]
        new_height = box.height.ravel()[0]
        new_dsize = (new_width, new_height)
        new = self.__class__(new_dsize, channels=channels, **self._kwargs)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('Nans._optimized_crop')
        return new

    def _optimized_warp(self, transform, dsize=None, **warp_kwargs):
        """
        Returns:
            DelayedImage
        """
        # Warping does nothing to nans, except maybe changing the dsize
        new = self.__class__(dsize, channels=self.channels, **self._kwargs)
        if TRACE_OPTIMIZE:
            new._opt_logs.append('Nans._optimized_warp')
        return new


class DelayedNodata(DelayedNans):
    """
    Constructs nan or masked array depending on what is needed

    Example:
        >>> from delayed_image.delayed_leafs import *  # NOQA
        >>> dsize = (307, 311)
        >>> self1 = DelayedNodata(dsize=dsize, channels='foo', nodata_method='float')
        >>> self2 = DelayedNodata(dsize=dsize, channels='foo', nodata_method='ma')
        >>> im1 = self1.finalize()
        >>> im2 = self2.finalize()
        >>> assert im1.dtype.kind == 'f'
        >>> assert not hasattr(im1, 'mask')
        >>> assert hasattr(im2, 'mask')
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedNans.__slots__

    def __init__(self, dsize=None, channels=None, nodata_method='float'):
        super().__init__(channels=channels, dsize=dsize)
        self.meta['nodata_method'] = nodata_method
        self._kwargs['nodata_method'] = nodata_method

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        shape = self.shape
        from delayed_image.helpers import _ensure_valid_shape
        shape = _ensure_valid_shape(shape)
        nodata_method = self.meta['nodata_method']
        if nodata_method == 'ma':
            # TODO: dtype should probably depend on what it will be combined
            # with?
            wrapped = np.empty(shape, dtype=np.uint8)
            final = np.ma.array(wrapped, dtype=np.uint8, mask=True)
        elif nodata_method is None or nodata_method in {'float', 'nan'}:
            final = np.full(shape, fill_value=np.nan)
        else:
            raise KeyError(nodata_method)
        return final


class DelayedIdentity(DelayedImageLeaf):
    """
    Returns an ndarray as-is

    Example:
        self = DelayedNans((10, 10), channel_spec.FusedChannelSpec.coerce('rgb'))
        region_slices = (slice(5, 10), slice(1, 12))
        delayed = self.crop(region_slices)

    Example:
        >>> from delayed_image import *  # NOQA
        >>> arr = kwimage.checkerboard()
        >>> self = DelayedIdentity(arr, channels='gray')
        >>> warp = self.warp({'scale': 1.07})
        >>> warp.optimize().finalize()
    """
    if delayed_base.USE_SLOTS:
        __slots__ = DelayedImageLeaf.__slots__ + ('data',)

    def __init__(self, data, channels=None, dsize=None):
        super().__init__(channels=channels)
        self.data = data
        self.meta['num_channels'] = kwimage.num_channels(data)
        if dsize is None:
            dsize = data.shape[0:2][::-1]
        self.meta['dsize'] = dsize

    def _finalize(self):
        """
        Returns:
            ArrayLike
        """
        return self.data

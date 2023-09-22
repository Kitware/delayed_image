"""
Stub that lays out the current structure of optimize calls.
"""
import ubelt as ub
import copy
from delayed_image.delayed_nodes import ImageOpsMixin
from delayed_image.delayed_nodes import DelayedConcat
from delayed_image.delayed_nodes import DelayedArray


class DelayedChannelConcat(ImageOpsMixin, DelayedConcat):

    def optimize(self):
        """
        Returns:
            DelayedImage
        """
        new_parts = [part.optimize() for part in self.parts]
        kw = ub.dict_isect(self.meta, ['dsize'])
        new = self.__class__(new_parts, **kw)
        return new


class DelayedImage(ImageOpsMixin, DelayedArray):
    """
    For the case where an array represents a 2D image with multiple channels
    """

    def _opt_push_under_concat(self):
        """
        Push this node under its child node if it is a concatenation operation
        """


class DelayedCrop(DelayedImage):

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
        if isinstance(new.subdata, DelayedCrop):
            new = new._opt_fuse_crops()

        if hasattr(new.subdata, '_optimized_crop'):
            # The subdata knows how to optimize itself wrt this node
            crop_kwargs = ub.dict_isect(self.meta, {'space_slice', 'chan_idxs'})
            new = new.subdata._optimized_crop(**crop_kwargs).optimize()
        if isinstance(new.subdata, DelayedWarp):
            new = new._opt_warp_after_crop()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedDequantize):
            new = new._opt_dequant_after_crop()
            new = new.optimize()
        if isinstance(new.subdata, DelayedChannelConcat):
            if isinstance(new, DelayedCrop):
                # We have to be careful if there we have band selection
                chan_idxs = new.meta.get('chan_idxs', None)
                space_slice = new.meta.get('space_slice', None)
                taken = new.subdata
                if chan_idxs is not None:
                    taken = new.subdata.take_channels(chan_idxs).optimize()
                if space_slice is not None:
                    taken = taken.crop(space_slice)._opt_push_under_concat().optimize()
                new = taken
            else:
                new = new._opt_push_under_concat().optimize()

        return new

    def _opt_fuse_crops(self):
        """
        Combine two consecutive crops into a single operation.
        """
        ...

    def _opt_warp_after_crop(self):
        """
        If the child node is a warp, move it after the crop.

        This is more efficient because:
            1. The crop is closer to the load.
            2. we are warping with less data.
        """
        ...

    def _opt_dequant_after_crop(self):
        """ Swap order so dequantize is after the crop """


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

    def optimize(self):
        """
        Returns:
            DelayedImage
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedWarp):
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
        elif isinstance(new.subdata, DelayedChannelConcat):
            new = new._opt_push_under_concat().optimize()
        elif hasattr(new.subdata, '_optimized_warp'):
            # The subdata knows how to optimize itself wrt a warp
            # (currently only exist for nans and constant leafs)
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
        return new

    def _opt_fuse_warps(self):
        """
        Combine two consecutive warps into a single operation.
        """

    def _opt_absorb_overview(self):
        """
        Remove the overview if we can get a higher resolution without it

        Given this warp node, if it has a scale component could undo an
        overview (i.e. the scale factor is greater than 2), we want to:

            1. determine if there is an overview deeper in the tree.
            2. remove that overview and that scale factor from this warp
            3. modify any intermediate nodes that will be changed by having the
               deeper overview removed.

        THIS FUNCTION IS COMPLEX AND MIGHT BE ABLE TO BE SIMPLIFIED
        """

    def _opt_split_warp_overview(self):
        """
        Split this node into a warp and an overview if possible
        """


class DelayedOverview(DelayedImage):
    def optimize(self):
        """
        Returns:
            DelayedImage
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()
        if isinstance(new.subdata, DelayedOverview):
            new = new._opt_fuse_overview()

        if new.meta['overview'] == 0:
            new = new.subdata
        elif isinstance(new.subdata, DelayedCrop):
            new = new._opt_crop_after_overview()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedWarp):
            new = new._opt_warp_after_overview()
            new = new.optimize()
        elif isinstance(new.subdata, DelayedDequantize):
            new = new._opt_dequant_after_overview()
            new = new.optimize()
        if isinstance(new.subdata, DelayedChannelConcat):
            new = new._opt_push_under_concat().optimize()
        return new

    def _opt_overview_as_warp(self):
        """
        Sometimes it is beneficial to replace an overview with a warp as an
        intermediate optimization step.
        """

    def _opt_crop_after_overview(self):
        """
        Given an outer overview and an inner crop, switch places. We want the
        overview to be as close to the load as possible.
        """

    def _opt_fuse_overview(self):
        ...

    def _opt_dequant_after_overview(self):
        ...

    def _opt_warp_after_overview(self):
        """
        Given an warp followed by an overview, move the warp to the outer scope
        such that the overview is first.
        """
        ...


class DelayedDequantize(DelayedImage):
    def optimize(self):
        """
        Returns:
            DelayedImage
        """
        new = copy.copy(self)
        new.subdata = self.subdata.optimize()

        if isinstance(new.subdata, DelayedDequantize):
            raise AssertionError('Dequantization is only allowed once')

        if isinstance(new.subdata, DelayedWarp):
            # Swap order so quantize is before the warp
            new = new._opt_dequant_before_other()
            new = new.optimize()

        if isinstance(new.subdata, DelayedChannelConcat):
            new = new._opt_push_under_concat().optimize()
        return new

    def _opt_dequant_before_other(self):
        ...

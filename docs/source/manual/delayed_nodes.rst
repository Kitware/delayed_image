``delayed_image.delayed_nodes``
===============================

.. currentmodule:: delayed_image.delayed_nodes

The :mod:`delayed_image.delayed_nodes` module defines the internal operation
nodes that sit between leaf data sources and finalized arrays. In practice,
this is the part of delayed-image that turns a chain of image operations into
an optimizable execution graph.

Leaf nodes such as :class:`delayed_image.delayed_leafs.DelayedLoad` and
:class:`delayed_image.delayed_leafs.DelayedNans` provide the source data.
This module provides the transform, crop, overview, dequantization, and
channel-composition nodes that are rearranged before execution.

Overview
--------

Most user-facing image operations are exposed by
:class:`~delayed_image.delayed_nodes.ImageOpsMixin` and implemented by
:class:`DelayedImage` subclasses.

The typical lifecycle is:

1. Start from a leaf node and call ``prepare()``.
2. Build a graph with operations like :meth:`DelayedImage.crop`,
   :meth:`DelayedImage.warp`, :meth:`DelayedImage.dequantize`, and
   :meth:`DelayedImage.get_overview`.
3. Call :meth:`DelayedImage.optimize` to rewrite the graph into a cheaper
   equivalent form.
4. Call :meth:`DelayedImage.finalize` to execute the optimized graph and
   materialize an array.

The key metadata carried through the graph is:

- ``dsize``: spatial size in ``(width, height)`` order.
- ``shape``: materialized shape in ``(height, width, channels)`` order.
- ``channels``: a :class:`delayed_image.channel_spec.FusedChannelSpec`
  describing semantic band names when available.

Quick Example
-------------

.. code-block:: python

   from delayed_image import DelayedLoad

   delayed = DelayedLoad.demo(channels='r|g|b').prepare()
   delayed = delayed.dequantize({'quant_max': 255, 'nodata': 0})
   delayed = delayed.warp({'scale': 0.5})
   delayed = delayed[10:90, 20:120]

   optimized = delayed.optimize()
   optimized.write_network_text()

   final = optimized.finalize()

Core Node Types
---------------

:class:`DelayedImage`
   Base class for image-like operation nodes. It carries ``dsize``,
   ``num_channels``, and ``channels`` metadata and provides transform queries
   such as :meth:`DelayedImage.get_transform_from_leaf`.

:class:`DelayedCrop`
   Integer pixel cropping plus optional channel selection. Adjacent crops are
   fused during optimization, and safe crops may be pushed closer to the data
   source.

:class:`DelayedWarp`
   Affine resampling node. This is where resize, scale, and general spatial
   transforms land. Optimization can fuse neighboring warps, factor large
   downsampling into overview reads, and absorb nearby overview nodes.

:class:`DelayedDequantize`
   Converts quantized integer imagery back to floating point and maps nodata
   to ``NaN`` where appropriate. The optimizer tries to move this later in the
   pipeline so earlier stages can operate on cheaper integer data.

:class:`DelayedOverview`
   Represents power-of-two downsampling. If the backing loader exposes true
   overviews, this can turn a large resample into a much cheaper pyramid read.

:class:`DelayedChannelConcat`
   Concatenates separate delayed images into a single multi-channel view.
   This is the main tool for multispectral or mixed-resolution fusion.

:class:`DelayedAsXarray`
   Finalization wrapper that returns an ``xarray.DataArray`` instead of a
   NumPy array.

Optimization Behavior
---------------------

The value of this module is mostly in its rewrite rules. The optimizer is not
just a no-op wrapper around the recorded operations.

Important rewrites include:

- Consecutive :class:`DelayedCrop` nodes are fused into one crop.
- Consecutive :class:`DelayedOverview` nodes are fused into a larger overview.
- Compatible :class:`DelayedWarp` nodes are fused into one affine transform.
- Crops, dequantization, and some warps are pushed under
  :class:`DelayedChannelConcat` so each branch can be optimized separately.
- Large downsampling warps are split into a real overview read plus a residual
  warp when the source provides overviews.
- Nearby overview nodes can be absorbed into a warp to simplify the graph.
- Some reordering is intentionally skipped for nearest-neighbor warps because
  exact pixel-center behavior is more important than a smaller graph.

This means ``optimize()`` is usually worth calling explicitly when you want to
inspect the graph, compare transforms, or reuse the same optimized pipeline
multiple times.

Working With Channels
---------------------

For single images, :meth:`DelayedImage.take_channels` can select bands by
index or by channel code. When a requested channel is missing, the operation
can fall back to :class:`DelayedChannelConcat` logic and synthesize ``NaN``
channels if requested.

For multi-branch stacks, :class:`DelayedChannelConcat` adds two important
capabilities:

- :meth:`DelayedChannelConcat.take_channels` can select and reorder channels
  across all branches.
- :meth:`DelayedChannelConcat.undo_warps` can return per-branch views in their
  original or partially unwarped coordinate systems, which is useful when
  aligned channels come from assets with different native resolutions.

Coordinate Introspection
------------------------

Several methods are useful when you need the graph structure itself, not just
the finalized pixels.

- :meth:`DelayedImage.get_transform_from_leaf` returns the affine transform
  from the underlying leaf into the current node's space.
- :meth:`ImageOpsMixin.get_transform_from` computes the transform between two
  delayed views that share a common leaf.
- :meth:`DelayedImage.evaluate` materializes the current node and wraps the
  result in a :class:`delayed_image.delayed_leafs.DelayedIdentity`, which is
  useful when you want to freeze an intermediate result.

Related API
-----------

The autogenerated reference page for the full module API is available at
:doc:`../auto/delayed_image.delayed_nodes`.

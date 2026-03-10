"""
Private type definitions for delayed_image.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias

import kwimage  # type: ignore[import-not-found]
import numpy as np

from delayed_image import delayed_base
from delayed_image.channel_spec import FusedChannelSpec

ArrayLike: TypeAlias = Any
BorderValueLike: TypeAlias = int | float | str | Sequence[int | float]
ChannelSpecLike: TypeAlias = FusedChannelSpec | str
ChannelSelectLike: TypeAlias = list[int] | slice | ChannelSpecLike | None
DSize: TypeAlias = tuple[int | None, int | None]
KnownDSize: TypeAlias = tuple[int, int]
MissingChannelPolicy: TypeAlias = Literal['return_nan', 'error']
PadLike: TypeAlias = int | Sequence[tuple[int, int]]
QuantizationSpec: TypeAlias = dict[str, Any]
SpaceSlice: TypeAlias = tuple[slice, slice]
WarpComponentKey: TypeAlias = Literal['offset', 'scale', 'shearx', 'theta']
WarpTransformLike: TypeAlias = np.ndarray | dict[str, Any] | kwimage.Affine | None


class DelayedImageLike(Protocol):
    dsize: DSize | None
    channels: FusedChannelSpec | None
    num_channels: int | None
    num_overviews: int | None
    meta: dict[str, Any]
    _opt_logs: list[str]

    def get_transform_from_leaf(self) -> kwimage.Affine: ...

    def _leaf_paths(self) -> Any: ...

    def _finalize(self) -> ArrayLike: ...

    def optimize(
        self, ctx: delayed_base.OptimizeContext | None = None
    ) -> DelayedImageLike: ...

    def crop(
        self,
        space_slice: SpaceSlice | None = None,
        chan_idxs: list[int] | None = None,
        clip: bool = True,
        wrap: bool = True,
        pad: PadLike = 0,
        lazy: bool = False,
    ) -> DelayedImageLike: ...

    def warp(
        self,
        transform: WarpTransformLike,
        dsize: KnownDSize | Literal['auto'] = 'auto',
        lazy: bool = False,
        **warp_kwargs: Any,
    ) -> DelayedImageLike: ...

    def take_channels(
        self,
        channels: ChannelSelectLike,
        lazy: bool = False,
        missing_channel_policy: MissingChannelPolicy = 'return_nan',
    ) -> DelayedImageLike: ...

    def undo_warp(
        self,
        remove: Sequence[WarpComponentKey] | None = None,
        retain: Sequence[WarpComponentKey] | set[WarpComponentKey] | None = None,
        squash_nans: bool = False,
        return_warp: bool = False,
    ) -> DelayedImageLike | tuple[DelayedImageLike, kwimage.Affine]: ...

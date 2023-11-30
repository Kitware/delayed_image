from typing import Tuple
from typing import List
import kwimage
from numpy import ndarray
from typing import Dict
from typing import Any
from _typeshed import Incomplete
from delayed_image import channel_spec
from delayed_image.delayed_base import DelayedNaryOperation, DelayedUnaryOperation

from delayed_image.channel_spec import FusedChannelSpec
from delayed_image.delayed_leafs import DelayedIdentity
from delayed_image.delayed_base import DelayedOperation

__docstubs__: str
TRACE_OPTIMIZE: int


class DelayedArray(DelayedUnaryOperation):

    def __init__(self, subdata: DelayedArray | None = None) -> None:
        ...

    def __nice__(self) -> str:
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
        ...


class DelayedStack(DelayedNaryOperation):

    def __init__(self, parts: List[DelayedArray], axis: int) -> None:
        ...

    def __nice__(self) -> str:
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
        ...


class DelayedConcat(DelayedNaryOperation):

    def __init__(self, parts: List[DelayedArray], axis: int) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
        ...


class DelayedFrameStack(DelayedStack):

    def __init__(self, parts: List[DelayedArray]) -> None:
        ...


class ImageOpsMixin:

    def crop(self,
             space_slice: Tuple[slice, slice] | None = None,
             chan_idxs: List[int] | None = None,
             clip: bool = True,
             wrap: bool = True,
             pad: int | List[Tuple[int, int]] = 0) -> DelayedImage:
        ...

    def warp(self,
             transform: ndarray | dict | kwimage.Affine,
             dsize: Tuple[int, int] | str = 'auto',
             **warp_kwargs) -> DelayedImage:
        ...

    def scale(self, scale, dsize: str = ..., **warp_kwargs):
        ...

    def resize(self, dsize, **warp_kwargs):
        ...

    def dequantize(self, quantization: Dict[str, Any]) -> DelayedDequantize:
        ...

    def get_overview(self, overview: int) -> DelayedOverview:
        ...

    def as_xarray(self) -> DelayedAsXarray:
        ...

    def get_transform_from(self, src: DelayedOperation) -> kwimage.Affine:
        ...


class DelayedChannelConcat(ImageOpsMixin, DelayedConcat):
    dsize: Tuple[int, int] | None
    num_channels: Incomplete

    def __init__(self,
                 parts: List[DelayedArray],
                 dsize: Tuple[int, int] | None = None) -> None:
        ...

    def __nice__(self) -> str:
        ...

    @property
    def channels(self) -> None | FusedChannelSpec:
        ...

    @property
    def shape(self) -> Tuple[int | None, int | None, int | None]:
        ...

    def optimize(self) -> DelayedImage:
        ...

    def take_channels(
        self, channels: List[int] | slice | channel_spec.FusedChannelSpec
    ) -> DelayedArray:
        ...

    def __getitem__(self, sl):
        ...

    @property
    def num_overviews(self) -> int:
        ...

    def as_xarray(self) -> DelayedAsXarray:
        ...

    def undo_warps(
        self,
        remove: List[str] | None = None,
        retain: List[str] | None = None,
        squash_nans: bool = False,
        return_warps: bool = False
    ) -> List[DelayedImage] | Tuple[List[DelayedImage] | List[kwimage.Affine]]:
        ...


class DelayedImage(ImageOpsMixin, DelayedArray):

    def __init__(self,
                 subdata: DelayedArray | None = None,
                 dsize: None | Tuple[int | None, int | None] = None,
                 channels: None | int | FusedChannelSpec = None) -> None:
        ...

    def __nice__(self) -> str:
        ...

    @property
    def shape(self) -> None | Tuple[int | None, int | None, int | None]:
        ...

    @property
    def num_channels(self) -> None | int:
        ...

    @property
    def dsize(self) -> None | Tuple[int | None, int | None]:
        ...

    @property
    def channels(self) -> None | FusedChannelSpec:
        ...

    @channels.setter
    def channels(self, channels) -> None | FusedChannelSpec:
        ...

    @property
    def num_overviews(self) -> int:
        ...

    def __getitem__(self, sl):
        ...

    def take_channels(
        self, channels: List[int] | slice | channel_spec.FusedChannelSpec
    ) -> DelayedCrop:
        ...

    def get_transform_from_leaf(self):
        ...

    def evaluate(self) -> DelayedIdentity:
        ...

    def undo_warp(self,
                  remove: List[str] | None = None,
                  retain: List[str] | None = None,
                  squash_nans: bool = False,
                  return_warp: bool = False):
        ...


class DelayedAsXarray(DelayedImage):

    def optimize(self) -> DelayedImage:
        ...


class DelayedWarp(DelayedImage):

    def __init__(self,
                 subdata: DelayedArray,
                 transform: ndarray | dict | kwimage.Affine,
                 dsize: Tuple[int, int] | str = 'auto',
                 antialias: bool = True,
                 interpolation: str = 'linear',
                 border_value: str = ...,
                 noop_eps: float = 0) -> None:
        ...

    @property
    def transform(self) -> kwimage.Affine:
        ...

    def optimize(self) -> DelayedImage:
        ...


class DelayedDequantize(DelayedImage):

    def __init__(self, subdata: DelayedArray, quantization: Dict) -> None:
        ...

    def optimize(self) -> DelayedImage:
        ...


class DelayedCrop(DelayedImage):
    channels: Incomplete

    def __init__(self,
                 subdata: DelayedArray,
                 space_slice: Tuple[slice, slice] | None = None,
                 chan_idxs: List[int] | None = None) -> None:
        ...

    def optimize(self) -> DelayedImage:
        ...


class DelayedOverview(DelayedImage):

    def __init__(self, subdata: DelayedArray, overview: int):
        ...

    @property
    def num_overviews(self) -> int:
        ...

    def optimize(self) -> DelayedImage:
        ...


class CoordinateCompatibilityError(ValueError):
    ...


class _InnerAccumSegment:

    def __init__(curr, comp) -> None:
        ...

    def add_inner(curr, inner, code) -> None:
        ...

    def get_indexer(curr):
        ...

    def get_subcomponent(curr, dsize):
        ...


def isinstance2(inst, cls: type) -> bool:
    ...

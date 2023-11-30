import kwimage
from os import PathLike
from typing import Tuple
from _typeshed import Incomplete
from delayed_image.delayed_nodes import DelayedImage

from delayed_image.channel_spec import FusedChannelSpec

__docstubs__: str


class DelayedImageLeaf(DelayedImage):

    def get_transform_from_leaf(self) -> kwimage.Affine:
        ...

    def optimize(self):
        ...


class DelayedLoad(DelayedImageLeaf):
    lazy_ref: Incomplete

    def __init__(self,
                 fpath: str | PathLike,
                 channels: int | str | FusedChannelSpec | None = None,
                 dsize: Tuple[int, int] | None = None,
                 nodata_method: str | None = None) -> None:
        ...

    @property
    def fpath(self):
        ...

    @classmethod
    def demo(DelayedLoad,
             key: str = 'astro',
             channels: str | None = None,
             dsize: None | Tuple[int, int] = None,
             nodata_method: str | None = None,
             overviews: None | int = None) -> DelayedLoad:
        ...

    def prepare(self) -> DelayedLoad:
        ...


class DelayedNans(DelayedImageLeaf):

    def __init__(self,
                 dsize: Incomplete | None = ...,
                 channels: Incomplete | None = ...) -> None:
        ...


class DelayedNodata(DelayedNans):

    def __init__(self,
                 dsize: Incomplete | None = ...,
                 channels: Incomplete | None = ...,
                 nodata_method: str = ...) -> None:
        ...


class DelayedIdentity(DelayedImageLeaf):
    data: Incomplete

    def __init__(self,
                 data,
                 channels: Incomplete | None = ...,
                 dsize: Incomplete | None = ...) -> None:
        ...

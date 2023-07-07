from typing import Dict
from typing import List
import networkx
from typing import Tuple
from typing import Any
from numpy.typing import ArrayLike
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator


class DelayedOperation(ub.NiceRepr):
    meta: Incomplete

    def __init__(self) -> None:
        ...

    def __nice__(self) -> str:
        ...

    def nesting(self) -> Dict[str, dict]:
        ...

    def as_graph(self, fields: str | List[str] = 'auto') -> networkx.DiGraph:
        ...

    def leafs(self) -> Generator[Tuple[DelayedOperation], None, None]:
        ...

    def print_graph(self,
                    fields: str = ...,
                    with_labels: bool = ...,
                    rich: str = ...,
                    vertical_chains: bool = ...) -> None:
        ...

    def write_network_text(self,
                           fields: str = ...,
                           with_labels: bool = ...,
                           rich: str = ...,
                           vertical_chains: bool = ...) -> None:
        ...

    @property
    def shape(self) -> None | Tuple[int | None, ...]:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def prepare(self) -> DelayedOperation:
        ...

    def finalize(self,
                 prepare: bool = True,
                 optimize: bool = True,
                 **kwargs) -> ArrayLike:
        ...

    def optimize(self) -> DelayedOperation:
        ...


class DelayedNaryOperation(DelayedOperation):
    parts: Incomplete

    def __init__(self, parts) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...


class DelayedUnaryOperation(DelayedOperation):
    subdata: Incomplete

    def __init__(self, subdata) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

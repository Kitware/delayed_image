from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


class BaseGlyphs:

    @classmethod
    def as_dict(cls):
        ...


class AsciiBaseGlyphs(BaseGlyphs):
    empty: str
    newtree_last: str
    newtree_mid: str
    endof_forest: str
    within_forest: str
    within_tree: str


class AsciiDirectedGlyphs(AsciiBaseGlyphs):
    last: str
    mid: str
    backedge: str
    vertical_edge: str


class AsciiUndirectedGlyphs(AsciiBaseGlyphs):
    last: str
    mid: str
    backedge: str
    vertical_edge: str


class UtfBaseGlyphs(BaseGlyphs):
    empty: str
    newtree_last: str
    newtree_mid: str
    endof_forest: str
    within_forest: str
    within_tree: str


class UtfDirectedGlyphs(UtfBaseGlyphs):
    last: str
    mid: str
    backedge: str
    vertical_edge: str


class UtfUndirectedGlyphs(UtfBaseGlyphs):
    last: str
    mid: str
    backedge: str
    vertical_edge: str


def generate_network_text(
        graph,
        with_labels: bool = ...,
        sources: Incomplete | None = ...,
        max_depth: Incomplete | None = ...,
        ascii_only: bool = ...,
        vertical_chains: bool = ...) -> Generator[Any, None, Any]:
    ...


def write_network_text(graph,
                       path: Incomplete | None = ...,
                       with_labels: bool = ...,
                       sources: Incomplete | None = ...,
                       max_depth: Incomplete | None = ...,
                       ascii_only: bool = ...,
                       end: str = ...,
                       vertical_chains: bool = ...) -> None:
    ...


def forest_str(graph,
               with_labels: bool = ...,
               sources: Incomplete | None = ...,
               write: Incomplete | None = ...,
               ascii_only: bool = ...):
    ...


def parse_network_text(lines):
    ...

"""Traversal helpers for delayed node trees."""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Tuple


ChildPath = Tuple[str, int | None]


def get_children(node) -> List[Tuple[ChildPath, object]]:
    """Return a list of child paths and nodes."""
    children: List[Tuple[ChildPath, object]] = []
    if hasattr(node, "subdata"):
        subdata = node.subdata
        if subdata is not None:
            children.append((("subdata", None), subdata))
    if hasattr(node, "parts"):
        parts = node.parts
        if parts is not None:
            for idx, part in enumerate(parts):
                children.append((("parts", idx), part))
    return children


def rebuild(node, new_children: Dict[ChildPath, object]):
    """Return a shallow copy of node with updated children."""
    if not new_children:
        return node
    new_node = copy.copy(node)
    if hasattr(new_node, "subdata") and ("subdata", None) in new_children:
        new_node.subdata = new_children[("subdata", None)]
    if hasattr(new_node, "parts"):
        parts = list(new_node.parts)
        for (kind, idx), child in new_children.items():
            if kind == "parts" and idx is not None:
                parts[idx] = child
        new_node.parts = parts
    return new_node

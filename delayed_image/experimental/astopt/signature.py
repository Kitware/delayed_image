"""Helpers for creating canonical signatures for delayed nodes."""

from __future__ import annotations

from typing import Any

import ubelt as ub


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "concise"):
        try:
            value = value.concise()
        except Exception:
            pass
    if hasattr(value, "spec"):
        try:
            value = value.spec
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _normalize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(val) for val in value]
    return value


def node_signature(node, child_signatures) -> str:
    """Return a hashable signature for a delayed node."""
    meta = getattr(node, "meta", {})
    normalized_meta = {key: _normalize_value(val) for key, val in meta.items()}
    payload = {
        "type": node.__class__.__name__,
        "meta": normalized_meta,
        "children": child_signatures,
    }
    try:
        return ub.hash_data(payload)
    except Exception:
        return ub.hash_data(ub.urepr(payload, sort=1))

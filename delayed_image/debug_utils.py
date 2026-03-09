"""
Helpers for opt-in low-level debugging in CI.
"""

from __future__ import annotations

import json
import os

import numpy as np

from delayed_image.constants import DEBUG_ARRAY_EVENTS


def debug_enabled():
    return bool(DEBUG_ARRAY_EVENTS)


def _array_info(arr):
    info = {
        'type': type(arr).__name__,
    }
    if hasattr(arr, 'dtype'):
        info['dtype'] = str(arr.dtype)
    if hasattr(arr, 'shape'):
        info['shape'] = tuple(map(int, arr.shape))
    if hasattr(arr, 'strides') and arr.strides is not None:
        info['strides'] = tuple(map(int, arr.strides))
    if hasattr(arr, 'flags'):
        info['c_contig'] = bool(arr.flags['C_CONTIGUOUS'])
        info['f_contig'] = bool(arr.flags['F_CONTIGUOUS'])
        info['owndata'] = bool(arr.flags['OWNDATA'])
        info['writeable'] = bool(arr.flags['WRITEABLE'])

    base_chain = []
    base = getattr(arr, 'base', None)
    seen = set()
    depth = 0
    while base is not None and id(base) not in seen and depth < 4:
        seen.add(id(base))
        base_info = {'type': type(base).__name__}
        if isinstance(base, np.ndarray):
            base_info['dtype'] = str(base.dtype)
            base_info['shape'] = tuple(map(int, base.shape))
            base_info['c_contig'] = bool(base.flags['C_CONTIGUOUS'])
            base_info['f_contig'] = bool(base.flags['F_CONTIGUOUS'])
            base_info['owndata'] = bool(base.flags['OWNDATA'])
        base_chain.append(base_info)
        base = getattr(base, 'base', None)
        depth += 1
    if base_chain:
        info['base_chain'] = base_chain
    return info


def debug_array_event(label, arr=None, **info):
    if not DEBUG_ARRAY_EVENTS:
        return
    payload = {
        'label': label,
        'pid': os.getpid(),
    }
    payload.update(info)
    if arr is not None:
        payload['array'] = _array_info(arr)
    text = (
        '[delayed_image.debug] '
        + json.dumps(payload, sort_keys=True, default=str)
        + '\n'
    )
    os.write(2, text.encode('utf8', errors='replace'))

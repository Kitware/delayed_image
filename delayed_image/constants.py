"""
Centralized runtime flags for hot-path conditionals.

Keep these as simple module-level integers so callers can use ``if FLAG:``
guards and avoid extra function-call overhead when the feature is disabled.
"""

from __future__ import annotations

import os

TRACE_OPTIMIZE = 0
IS_DEVELOPING = 0
DEBUG_ARRAY_EVENTS = int(
    bool(os.environ.get('DELAYED_IMAGE_DEBUG_SEGFAULT', ''))
)

# The dataset-level GDAL fast path is currently disabled while we debug
# CI-specific crashes. Keep this as a top-level flag so the branch remains
# cheap in hot paths.
GDAL_FAST_PATH = 0


__all__ = [
    'TRACE_OPTIMIZE',
    'IS_DEVELOPING',
    'DEBUG_ARRAY_EVENTS',
    'GDAL_FAST_PATH',
]

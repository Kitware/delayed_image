#!/usr/bin/env python3
"""
Minimal repro for a dtype-specific OpenCV warp regression.

This script exercises ``cv2.warpAffine`` directly without kwimage or
delayed_image. On the affected runtime stack, ``float64`` inputs with nearest
interpolation behave as if the matrix direction is flipped, while ``float32``
and ``uint8`` behave normally.

Example:
    python dev/repro_opencv_warp_affine_float64_nearest.py
"""

from __future__ import annotations

import cv2
import numpy as np


def summarize_warp(src: np.ndarray, matrix: np.ndarray, interp: int) -> dict:
    warped = cv2.warpAffine(
        src,
        matrix,
        dsize=(52, 51),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan if src.dtype.kind == "f" else 0,
    )
    if warped.dtype.kind == "f":
        finite = np.isfinite(warped)
    else:
        finite = np.ones(warped.shape, dtype=bool)
    uniq = np.unique(warped[finite]) if finite.any() else np.array([])
    return {
        "finite_ratio": float(finite.mean()),
        "unique_count": int(len(uniq)),
        "unique_head": uniq[:8].tolist(),
    }


def main() -> None:
    raw = np.linspace(0, 1, 36).reshape(6, 6)
    forward = np.array([[8.6, 0, 0], [0, 8.5, 0]], dtype=np.float64)
    inverse = np.array([[1 / 8.6, 0, 0], [0, 1 / 8.5, 0]], dtype=np.float64)

    print("opencv", cv2.__version__)
    print("numpy", np.__version__)
    print()

    for interp_name in ["INTER_NEAREST", "INTER_LINEAR"]:
        interp = getattr(cv2, interp_name)
        print(interp_name)
        for src_dtype in [np.uint8, np.float32, np.float64]:
            if src_dtype == np.uint8:
                src = (raw * 255).astype(src_dtype)
            else:
                src = raw.astype(src_dtype)
            forward_info = summarize_warp(src, forward, interp)
            inverse_info = summarize_warp(src, inverse, interp)
            info = {
                'src_dtype': np.dtype(src_dtype).name,
                'forward': forward_info,
                'inverse': inverse_info,
            }
            import ubelt as ub
            import rich
            rich.print(f'info = {ub.urepr(info, nl=2)}')
        print()

    f64_nearest_forward = summarize_warp(raw.astype(np.float64), forward, cv2.INTER_NEAREST)
    f64_nearest_inverse = summarize_warp(raw.astype(np.float64), inverse, cv2.INTER_NEAREST)
    bug_present = (
        f64_nearest_forward["unique_count"] < f64_nearest_inverse["unique_count"]
        and f64_nearest_forward["finite_ratio"] < f64_nearest_inverse["finite_ratio"]
    )
    print("float64 nearest regression present:", bug_present)


if __name__ == "__main__":
    main()

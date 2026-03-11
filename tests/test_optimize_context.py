from __future__ import annotations

import warnings

import numpy as np
import pytest

import delayed_image


def _finalize_ignoring_warnings(node):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return node.finalize()


def _require_warp_backend():
    from kwimage import im_transform

    backend = im_transform._default_backend()
    if backend == 'skimage':
        pytest.skip('kwimage warp/imresize backend is unavailable')


def test_optimize_idempotence():
    _require_warp_backend()
    rng = np.random.default_rng(0)
    data = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    base = delayed_image.DelayedIdentity(data, channels='r|g|b')
    base.meta['num_overviews'] = 1
    quantization = {'quant_max': 255, 'nodata': 0}

    node = base.dequantize(quantization)
    node = node.warp(
        {'scale': 1.1, 'offset': (2, -1)},
        interpolation='nearest',
        antialias=False,
    )
    node = node.crop((slice(2, 24), slice(3, 25)))
    node = node.get_overview(1)

    opt1 = node.optimize()
    opt2 = opt1.optimize()

    assert opt1.nesting() == opt2.nesting()
    final1 = _finalize_ignoring_warnings(opt1)
    final2 = _finalize_ignoring_warnings(opt2)
    assert np.allclose(final1, final2, equal_nan=True)


def test_repeated_optimize_equivalence():
    _require_warp_backend()
    rng = np.random.default_rng(1)
    data = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
    base = delayed_image.DelayedIdentity(data, channels='r|g|b')
    quantization = {'quant_max': 255, 'nodata': 0}

    node = base.warp(
        {'scale': (1.2, 0.9), 'theta': 0.05}, interpolation='linear'
    )
    node = node.crop((slice(4, 40), slice(5, 41)))
    node = node.dequantize(quantization)

    opt1 = node.optimize()
    opt2 = node.optimize()

    final_orig = _finalize_ignoring_warnings(node)
    final1 = _finalize_ignoring_warnings(opt1)
    final2 = _finalize_ignoring_warnings(opt2)

    assert np.allclose(final1, final2, equal_nan=True)
    assert np.allclose(final_orig, final1, equal_nan=True)


def test_randomized_tree_finalize_equivalence():
    _require_warp_backend()
    rng = np.random.default_rng(2)
    data = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    base = delayed_image.DelayedIdentity(data, channels='r|g|b')
    base.meta['num_overviews'] = 1
    quantization = {'quant_max': 255, 'nodata': 0}

    node = base.dequantize(quantization)
    node = node.get_overview(1)
    node = node.scale(
        rng.uniform(0.6, 1.4),
        dsize='auto',
        interpolation='linear',
        antialias=True,
    )
    node = node.warp(
        {
            'scale': (rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3)),
            'offset': (rng.uniform(-5, 5), rng.uniform(-5, 5)),
            'theta': rng.uniform(-0.2, 0.2),
        },
        dsize='auto',
        interpolation='nearest',
    )

    w, h = node.dsize
    y0 = rng.integers(0, max(1, h // 4))
    y1 = rng.integers(max(y0 + 1, h // 2), h)
    x0 = rng.integers(0, max(1, w // 4))
    x1 = rng.integers(max(x0 + 1, w // 2), w)
    node = node.crop((slice(int(y0), int(y1)), slice(int(x0), int(x1))))

    final_raw = _finalize_ignoring_warnings(node)
    final_opt = _finalize_ignoring_warnings(node.optimize())
    assert np.allclose(final_raw, final_opt, equal_nan=True)


def test_optimize_preserves_metadata(tmp_path):
    _require_warp_backend()
    rng = np.random.default_rng(3)
    data = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    fpath = tmp_path / 'meta.png'
    import kwimage

    kwimage.imwrite(str(fpath), data)
    base = delayed_image.DelayedLoad(
        fpath, channels='r|g|b', nodata_method='float'
    ).prepare()
    quantization = {'quant_max': 255, 'nodata': 0}

    node = base.dequantize(quantization)
    node = node.warp(
        {'scale': 1.3, 'offset': (2, -1)},
        interpolation='nearest',
        antialias=False,
        border_value=0,
        dsize='auto',
    )
    node = node.crop((slice(5, 40), slice(4, 50)))

    opt = node.optimize()

    assert opt.channels == node.channels
    assert opt.dsize == node.dsize

    warp_nodes = [
        n
        for _, n in opt._traverse()
        if isinstance(n, delayed_image.DelayedWarp)
    ]
    assert warp_nodes, 'optimized graph should retain a warp'
    warp = warp_nodes[0]
    assert warp.meta['interpolation'] == 'nearest'
    assert warp.meta['antialias'] is False

    load_nodes = [
        n
        for _, n in opt._traverse()
        if isinstance(n, delayed_image.DelayedLoad)
    ]
    assert load_nodes, 'optimized graph should retain a load node'
    assert load_nodes[0].meta['nodata_method'] == 'float'


def test_linear_crop_after_warp_rewrite_equivalence():
    _require_warp_backend()
    rng = np.random.default_rng(4)
    data = rng.random((96, 96, 3), dtype=np.float32)
    base = delayed_image.DelayedIdentity(data, channels='r|g|b')

    node = base.warp({'scale': 0.75}, interpolation='linear')
    node = node.crop((slice(8, 60), slice(5, 58)))
    node = node.warp(
        {'scale': 1.35, 'offset': (2.5, -3.5)}, interpolation='linear'
    )
    node = node.crop((slice(0, 40), slice(0, 40)))
    node = node.take_channels([0, 1])

    opt = node.optimize()
    final_raw = _finalize_ignoring_warnings(node)
    final_opt = _finalize_ignoring_warnings(opt)

    assert np.allclose(final_raw, final_opt, equal_nan=True)

    warp_nodes = [
        n
        for _, n in opt._traverse()
        if isinstance(n, delayed_image.DelayedWarp)
    ]
    assert len(warp_nodes) == 1

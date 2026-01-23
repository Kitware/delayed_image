import numpy as np
import pytest

from delayed_image import DelayedIdentity


def _make_base(seed=0, dsize=(64, 64)):
    rng = np.random.RandomState(seed)
    data = rng.randint(0, 255, size=(dsize[1], dsize[0], 3), dtype=np.uint8)
    return DelayedIdentity(data, channels='r|g|b')


def test_ast_optimize_equivalence_simple(optimize_pair):
    base = _make_base()
    quantization = {'quant_max': 255, 'nodata': 0}
    node = base.dequantize(quantization)
    node = node.crop((slice(2, 60), slice(4, 66)))
    node = node.take_channels('r|b')

    legacy, ast = optimize_pair(node)

    legacy_final = legacy.finalize()
    ast_final = ast.finalize()

    assert legacy.dsize == ast.dsize
    np.testing.assert_allclose(legacy_final, ast_final, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ast_optimize_equivalence_random(seed, optimize_pair):
    rng = np.random.RandomState(seed)
    base = _make_base(seed=seed)
    quantization = {'quant_max': 255, 'nodata': 0}

    node = base
    used_dequant = False
    for _ in range(5):
        op = rng.choice(["crop", "dequant", "channels"])
        if op == "crop":
            x0 = rng.randint(0, 20)
            y0 = rng.randint(0, 20)
            node = node.crop((slice(y0, y0 + 32), slice(x0, x0 + 32)))
        elif op == "channels":
            node = node.take_channels('r|g')
        elif op == "dequant" and not used_dequant:
            node = node.dequantize(quantization)
            used_dequant = True

    legacy, ast = optimize_pair(node)
    np.testing.assert_allclose(legacy.finalize(), ast.finalize(), atol=1e-6)


def test_optimize_idempotent(optimize_func):
    base = _make_base()
    quantization = {'quant_max': 255, 'nodata': 0}
    node = base.dequantize(quantization).crop((slice(0, 40), slice(0, 40)))
    node = node.take_channels('r|b')

    opt1 = optimize_func(node)
    opt2 = optimize_func(opt1)

    np.testing.assert_allclose(opt1.finalize(), opt2.finalize(), atol=1e-6)

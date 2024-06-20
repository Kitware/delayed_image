

def test_subchannel_select_with_overviews_case1():
    """
    This reproduces a bug in version < 0.2.8 with the exact operation tree that
    caused it in production.
    """
    import delayed_image
    try:
        import osgeo
    except ImportError:
        import pytest
        pytest.skip('test has overviews')

    leaf = delayed_image.DelayedLoad.demo(channels='r|g|b', overviews=3)
    leaf.prepare()

    class meta_mkslice:
        def __getitem__(self, index):
            return index
    mkslice = meta_mkslice()

    quantization = {
        'orig_dtype': 'int16',
        'orig_min': -94,
        'orig_max': 6040,
        'quant_min': 0,
        'quant_max': 32767,
        'nodata': -9999,
    }

    node = leaf
    node = node.crop(mkslice[0:387, 0:387], chan_idxs=[0, 2])
    node = node.dequantize(quantization)
    node = node.warp({'type': 'affine'}, dsize=(384, 384))
    node = delayed_image.DelayedChannelConcat([node])
    node = node.warp({'type': 'affine', 'scale': 1 / 3}, dsize=(128, 128))

    print(chr(10) + 'Before Optimization:')
    node.write_network_text()

    optimized = node.optimize()

    print(chr(10) + 'After Optimization:')
    optimized.write_network_text()

    im1 = node.finalize(optimize=False)
    im2 = optimized.finalize()

    assert node.shape[2] == 2
    assert optimized.shape[2] == 2

    assert im1.shape[2] == 2
    assert im2.shape[2] == 2


def test_subchannel_select_with_overviews_case2():
    """
    This reproduces a bug in version < 0.2.8 with a minimal example
    """
    import delayed_image
    try:
        import osgeo
    except ImportError:
        import pytest
        pytest.skip('test has overviews')

    leaf = delayed_image.DelayedLoad.demo(channels='r|g|b', overviews=3)
    leaf.prepare()

    node = leaf
    node = node.crop(chan_idxs=[0, 2])
    node = node.warp({'type': 'affine', 'scale': 1 / 3}, dsize=(128, 128))

    print(chr(10) + 'Before Optimization:')
    node.write_network_text()

    optimized = node.optimize()

    print(chr(10) + 'After Optimization:')
    optimized.write_network_text()

    im1 = node.finalize(optimize=False)
    im2 = optimized.finalize()

    assert node.shape[2] == 2
    assert optimized.shape[2] == 2

    assert im1.shape[2] == 2
    assert im2.shape[2] == 2

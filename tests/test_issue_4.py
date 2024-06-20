def test_issue4():
    """
    The symptom is given this tree:

        ╙── Warp dsize=(225,225),transform={offset=(0.0000,21.0000)}
            ╽
            Crop dsize=(0,204.0000),space_slice=(slice(0,204.0,None),slice(219,219,None))
            ╽
            Warp dsize=(219,219),transform={offset=(-0.0000,0.0000),scale=(0.9992,0.9992)}
            ╽
            ChannelConcat axis=2
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=blue,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_blue.tif
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=green,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_green.tif
            └─╼ Warp dsize=(219,219),transform={}
                ╽
                Load channels=red,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_red.tif

    Optimize resulted in:

        ~/.pyenv/versions/3.11.2/lib/python3.11/site-packages/delayed_image/helpers.py in _swap_warp_after_crop(root_region_bounds, tf_leaf_to_root)
            157
            158         # TODO: test the case where old_w or old_h are zero
        --> 159         padw = int(np.ceil(leaf_w / old_w))
            160         padh = int(np.ceil(leaf_h / old_h))
            161     else:

        OverflowError: cannot convert float infinity to integer



    The sequence of operations to build the tree was:

            delayed_frame = coco_img.imdelay(
                channels=request_chanspec, space=space,
                interpolation=interpolation,
                nodata_method=nodata,
                antialias=antialias
            )

        In [175]: delayed_frame.write_network_text()
        ╙── Warp dsize=(219,219),transform={offset=(-0.0000,0.0000),scale=(0.9992,0.9992)}
            ╽
            ChannelConcat axis=2
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=blue,dsize=(219,219),fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_blue.tif
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=green,dsize=(219,219),fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_green.tif
            └─╼ Warp dsize=(219,219),transform={}
                ╽
                Load channels=red,dsize=(219,219),fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_red.tif


        In [178]: requested_space_slice
        Out[178]: (slice(-21.0, 204.0, None), slice(219, 444.0, None))

        In [179]: space_pad
        Out[179]: [(0, 0), (0, 0)]

        delayed_crop = delayed_frame.crop(requested_space_slice,
                                          clip=False, wrap=False,
                                          pad=space_pad)
        delayed_crop = delayed_crop.prepare()

        In [177]: delayed_crop.write_network_text()
        ╙── Warp dsize=(225,225),transform={offset=(0.0000,21.0000)}
            ╽
            Crop dsize=(0,204.0000),space_slice=(slice(0,204.0,None),slice(219,219,None))
            ╽
            Warp dsize=(219,219),transform={offset=(-0.0000,0.0000),scale=(0.9992,0.9992)}
            ╽
            ChannelConcat axis=2
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=blue,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_blue.tif
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=green,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_green.tif
            └─╼ Warp dsize=(219,219),transform={}
                ╽
                Load channels=red,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_red.tif

        delayed_crop = delayed_crop.optimize()

        Error

    Note:
        The issue seems to be that the crop is ouside the bounds of the image,
        so we just need to ensure the divide by zero doesn't happen.

    """
    from delayed_image import DelayedChannelConcat
    import delayed_image
    import kwimage
    import numpy as np

    try:
        import osgeo
    except ImportError:
        import pytest
        pytest.skip('test uses overviews. needs osgeo.gdal')

    r = delayed_image.DelayedLoad.demo(channels='r', overviews=6, dsize=(219, 219))
    g = delayed_image.DelayedLoad.demo(channels='g', overviews=6, dsize=(219, 219))
    b = delayed_image.DelayedLoad.demo(channels='b', overviews=6, dsize=(219, 219))

    concat = DelayedChannelConcat([r.warp({}), g.warp({}), b.warp({})])

    mat = kwimage.Affine(np.array([[ 9.99169700e-01,  0.00000000e+00, -5.82076609e-11],
                                   [ 0.00000000e+00,  9.99169700e-01,  2.91038305e-11],
                                   [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]))
    delayed_frame = concat.warp(mat)
    delayed_frame.print_graph()

    requested_space_slice = (slice(-21.0, 204.0, None), slice(219, 444.0, None))
    space_pad = [(0, 0), (0, 0)]
    delayed_crop = delayed_frame.crop(requested_space_slice,
                                      clip=False, wrap=False,
                                      pad=space_pad)
    delayed_crop = delayed_crop.prepare()

    delayed_crop.print_graph()
    optimized = delayed_crop.optimize()
    optimized.print_graph()
    assert optimized.dsize == (225, 225)


def test_clipped_negative_slice():
    import delayed_image
    from delayed_image.helpers import mkslice
    base = delayed_image.DelayedLoad.demo(dsize=(256, 256))
    slices = mkslice[-10:216, 0:256]
    cropped = base.crop(slices)
    assert cropped.dsize == (256, 0)
    assert cropped.optimize().dsize == (256, 0)


def test_oob_crop_after_load():
    import delayed_image
    import ubelt as ub
    from delayed_image.helpers import mkslice
    base = delayed_image.DelayedLoad.demo(dsize=(256, 256))
    slices = mkslice[300:500, 400:500]

    pad = [(0, 0), (0, 0)]
    variants = {}
    variants['v1'] = base.crop(slices)
    variants['v2'] = base.crop(slices, wrap=False, clip=False)
    variants['v3'] = base.crop(slices, wrap=False, clip=False, pad=pad)

    outputs = {}
    for key, orig in variants.items():
        print('----------')
        print('key = {}'.format(ub.urepr(key, nl=1)))
        orig.print_graph()
        orig.prepare()
        opt = orig.optimize()
        opt.print_graph()
        outputs[key] = opt
        print('----------')

    assert outputs['v1'].dsize == (0, 0)
    assert outputs['v2'].dsize == (100, 200)
    assert outputs['v3'].dsize == (100, 200)


def test_oob_crop_after_warp():
    """
    Like test_oob_crop_after_load, but adds in a warp before the slices that
    triggered errors the previous test did not.
    """
    import delayed_image
    from delayed_image.helpers import mkslice
    base = delayed_image.DelayedLoad.demo(dsize=(256, 256))
    base = base.warp({'scale': 1.01})
    slices = mkslice[300:500, 400:500]

    pad = [(0, 0), (0, 0)]
    variants = {}
    variants['v1'] = base.crop(slices)
    variants['v2'] = base.crop(slices, wrap=False, clip=False)
    variants['v3'] = base.crop(slices, wrap=False, clip=False, pad=pad)

    import ubelt as ub
    errors = []
    outputs = {}
    for key, orig in variants.items():
        print('----------')
        print('key = {}'.format(ub.urepr(key, nl=1)))
        orig.print_graph()
        orig.prepare()
        try:
            opt = orig.optimize()
        except Exception as ex:
            print('ex = {}'.format(ub.urepr(ex, nl=1)))
            errors.append((key, ex))
            ...
        opt.print_graph()
        outputs[key] = opt
        print('----------')
    print('errors = {}'.format(ub.urepr(errors, nl=1)))

    assert outputs['v1'].dsize == (0, 0)
    assert outputs['v2'].dsize == (100, 200)
    assert outputs['v3'].dsize == (100, 200)


def test_oob_crop_after_warp_with_overviews():
    """
    Like test_oob_crop_after_load, but adds in a warp before the slices that
    triggered errors the previous test did not.
    """
    import delayed_image
    from delayed_image.helpers import mkslice

    try:
        import osgeo
    except ImportError:
        import pytest
        pytest.skip('test uses overviews. needs osgeo.gdal')

    base = delayed_image.DelayedLoad.demo(dsize=(256, 256), overviews=3)
    base = base.warp({'scale': 0.1})
    slices = mkslice[300:500, 400:500]

    pad = [(0, 0), (0, 0)]
    variants = {}
    variants['v1'] = base.crop(slices)
    variants['v2'] = base.crop(slices, wrap=False, clip=False)
    variants['v3'] = base.crop(slices, wrap=False, clip=False, pad=pad)

    import ubelt as ub
    errors = []
    outputs = {}
    for key, orig in variants.items():
        print('----------')
        print('key = {}'.format(ub.urepr(key, nl=1)))
        orig.print_graph()
        orig.prepare()
        try:
            opt = orig.optimize()
        except Exception as ex:
            print('ex = {}'.format(ub.urepr(ex, nl=1)))
            errors.append((key, ex))
            ...
        opt.print_graph()
        outputs[key] = opt
        print('----------')
    print('errors = {}'.format(ub.urepr(errors, nl=1)))

    assert outputs['v1'].dsize == (0, 0)
    assert outputs['v2'].dsize == (100, 200)
    assert outputs['v3'].dsize == (100, 200)


def test_both_total_negative_slice():
    import delayed_image
    try:
        import osgeo
    except ImportError:
        import pytest
        pytest.skip('test uses overviews. needs osgeo.gdal')
    base = delayed_image.DelayedLoad.demo(dsize=(128, 128), overviews=3)
    base = base.warp({'scale': 2.0})
    slices = (slice(-173, -60, None), slice(-123, -10, None))
    pad = [(0, 0), (0, 0)]
    crop = base.crop(slices, wrap=False, clip=False, pad=pad)
    crop.print_graph()
    opt = crop.optimize()
    assert crop.dsize == opt.dsize
    opt.print_graph()

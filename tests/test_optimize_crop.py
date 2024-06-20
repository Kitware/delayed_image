def test_optimize_crop_without_clip_reproduction():
    """
    There was an issue where a non-clipped crop would optimize it with a clip.
    This reproduces the issue exactly as it was originally seen.

    Original Issue Graph:

        Input:
        ╙── Warp dsize=(416,416),transform={scale=0.2500}
            ╽
            ChannelConcat axis=2
            ╽
            Warp dsize=(1664,1664),transform={}
            ╽
            Crop dsize=(1664.0000,1428.0000),space_slice=(slice(1596.0,3024,None),slice(1512.5,3176.5,None))
            ╽
            Load channels=red|green|blue,dsize=(4032,3024),nodata_method=float,num_overviews=3,fname=PXL_20210127_145659342.jpg

        Optimized:
        ╙── ChannelConcat axis=2
            ╽
            Warp dsize=(417,357),transform={offset=(-0.1250,0.0000)}
            ╽
            Crop dsize=(417,357),space_slice=(slice(399,756,None),slice(378,795,None))
            ╽
            Overview dsize=(1008,756),overview=2
            ╽
            Load channels=red|green|blue,dsize=(4032,3024),nodata_method=float,num_overviews=3,fname=PXL_20210127_145659342.jpg
    """
    import delayed_image
    import kwimage
    try:
        import osgeo
    except ImportError:
        import pytest
        pytest.skip()
    fpath = kwimage.grab_test_image_fpath(overviews=3, dsize=(4032, 3024))
    base = delayed_image.DelayedLoad(fpath, channels='r|g|b').prepare()
    space_slice = (slice(1596.0, 3024), slice(1512.5, 3176.5))
    delayed = base.crop(space_slice, clip=False, wrap=False)
    delayed = delayed_image.DelayedChannelConcat([delayed])
    delayed = delayed.warp({'scale': 0.25}, dsize=(416, 416))
    delayed.print_graph()

    delayed_image.delayed_nodes.TRACE_OPTIMIZE = 1

    optimize = delayed.optimize()
    optimize.print_graph()

    if delayed_image.delayed_nodes.TRACE_OPTIMIZE:
        print('Opt Logs1')
        chain1 = [n for _, n in optimize._traverse()]
        for n in chain1:
            print(n._opt_logs)

    assert delayed.dsize == (416, 416), ('original image has a specific size')
    assert optimize.dsize == (416, 416), ('optimization should keep that size')


def test_optimize_crop_without_clip_simplified():
    """
    This reproduces a simplified minimal version of the issue
    """
    import delayed_image
    import kwimage
    try:
        import osgeo
    except ImportError:
        import pytest
        pytest.skip()
    fpath = kwimage.grab_test_image_fpath(overviews=0, dsize=(416, 416))
    base = delayed_image.DelayedLoad(fpath, channels='r|g|b').prepare()
    space_slice = (slice(0, 357), slice(0, 416))
    delayed = base.crop(space_slice, clip=False, wrap=False)
    delayed = delayed.warp({'scale': 0.25}, dsize=(416, 416))
    delayed.print_graph()

    optimize = delayed.optimize()
    optimize.print_graph()

    assert delayed.dsize == (416, 416), ('original image has a specific size')
    assert optimize.dsize == (416, 416), ('optimization should keep that size')


def test_optimize_crop_without_clip_minimal():
    """
    Minimal operations that caused the issue
    """
    import delayed_image
    import kwimage
    try:
        import osgeo
    except ImportError:
        import pytest
        pytest.skip()
    fpath = kwimage.grab_test_image_fpath(overviews=3, dsize=(4032, 3024))
    base = delayed_image.DelayedLoad(fpath, channels='r|g|b').prepare()
    delayed = base
    delayed = delayed.warp({'scale': 0.25}, dsize=(416, 416))
    delayed.print_graph()

    delayed_image.delayed_nodes.TRACE_OPTIMIZE = 1

    optimize = delayed.optimize()
    optimize.print_graph()

    if delayed_image.delayed_nodes.TRACE_OPTIMIZE:
        print('Opt Logs1')
        chain1 = [n for _, n in optimize._traverse()]
        for n in chain1:
            print(n._opt_logs)

    assert delayed.dsize == (416, 416), ('original image has a specific size')
    assert optimize.dsize == (416, 416), ('optimization should keep that size')


def test_delayed_load_int_nodata():
    import ubelt as ub
    dpath = ub.Path.appdir('delayed_image/test/nodata_test').ensuredir()

    try:
        from osgeo import gdal  # NOQA
    except ImportError:
        import pytest
        pytest.skip('requires gdal')

    # Create an image on disk with encoded integer nodata.
    import kwimage
    im = kwimage.grab_test_image()
    im = kwimage.ensure_uint255(im)

    im[0, :] = 0
    fpath = dpath / 'nodata_int.tiff'

    kwimage.imwrite(fpath, im, nodata_value=0, overviews=3, backend='gdal')

    new_im_int = kwimage.imread(fpath, nodata_method='ma')
    new_im_flt = kwimage.imread(fpath, nodata_method='float')

    import numpy as np
    assert np.all(new_im_int.mask == np.isnan(new_im_flt))
    assert np.all(new_im_int.mask == (im == 0))

    import delayed_image
    delayed = delayed_image.DelayedLoad(fpath)

    final_nod = delayed.finalize(nodata_method='ma')
    final_nan = delayed.finalize(nodata_method='float')

    assert np.all(np.isnan(final_nan) == final_nod.mask)


def test_delayed_cat_int_nodata():
    import ubelt as ub
    dpath = ub.Path.appdir('delayed_image/test/nodata_test').ensuredir()

    try:
        from osgeo import gdal  # NOQA
    except ImportError:
        import pytest
        pytest.skip('requires gdal')

    # Create an image on disk with encoded integer nodata.
    import kwimage
    im = kwimage.grab_test_image()
    im = kwimage.ensure_uint255(im)

    im[0, :] = 0
    fpath = dpath / 'nodata_int.tiff'

    kwimage.imwrite(fpath, im, nodata_value=0, overviews=3, backend='gdal')

    new_im_int = kwimage.imread(fpath, nodata_method='ma')
    new_im_flt = kwimage.imread(fpath, nodata_method='float')

    import numpy as np
    assert np.all(new_im_int.mask == np.isnan(new_im_flt))
    assert np.all(new_im_int.mask == (im == 0))

    import delayed_image
    delayed = delayed_image.DelayedLoad(fpath)
    delayed.print_graph()
    delayed.prepare()
    delayed.print_graph()

    nodata1 = delayed_image.DelayedNodata(dsize=delayed.dsize, channels=1)
    nodata1.print_graph()

    nodata2 = delayed_image.DelayedNodata(dsize=delayed.dsize, channels=1, nodata_method='ma')
    purenans = delayed_image.DelayedNans(dsize=delayed.dsize, channels=1)

    cat1 = delayed_image.DelayedChannelConcat([delayed, nodata1])
    cat2 = delayed_image.DelayedChannelConcat([delayed, nodata2])
    cat3 = delayed_image.DelayedChannelConcat([delayed, purenans])

    cat1.print_graph(fields='all')
    cat2.print_graph(fields='all')
    cat3.print_graph(fields='all')

    result1a = cat1.finalize(nodata_method='nan')
    result2a = cat2.finalize(nodata_method='nan')

    result1b = cat1.finalize(nodata_method='ma')
    result2b = cat2.finalize(nodata_method='ma')
    result3 = cat3.finalize()

    print(f'result1a.dtype={result1a.dtype}')
    print(f'result2a.dtype={result2a.dtype}')
    print(f'result1b.dtype={result1b.dtype}')
    print(f'result2b.dtype={result2b.dtype}')
    print(f'result3.dtype={result3.dtype}')

    # TODO / FIXME: Currently the user needs to correctly ensure the nodes in
    # the tree have compatible nodata_method values, but it would be better to
    # be able to let the user specify it at finalize time.

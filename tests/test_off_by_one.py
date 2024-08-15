
def test_off_by_one_with_upscale():
    import delayed_image
    delayed = delayed_image.DelayedLoad.demo(key='astro').prepare()

    x = delayed.scale(2.0001)[0:4, 0:4]
    data1 = x.finalize(optimize=False, nodata_method='float')
    data2 = x.finalize(nodata_method='float')

    import kwarray
    assert kwarray.ArrayAPI.coerce('numpy').array_equal(data2, data1, equal_nan=True)
    if 0:
        import kwplot
        kwplot.autompl()
        kwplot.plt.ion()
        kwplot.imshow(data1, pnum=(1, 2, 1))
        kwplot.imshow(data2, pnum=(1, 2, 2))


def test_off_by_one_with_multi_scale():
    import delayed_image
    import numpy as np
    delayed = delayed_image.DelayedLoad.demo(key='astro').prepare()

    for s in np.linspace(0.4, 2.5, 100).tolist() + [0.5, 1.0, 2.0]:
        x = delayed.scale(s)[0:4, 0:4]
        data1 = x.finalize(optimize=False, nodata_method='float')
        data2 = x.finalize(nodata_method='float')
        # assert np.all(data2 == data1)
        import kwarray
        assert kwarray.ArrayAPI.coerce('numpy').array_equal(data2, data1, equal_nan=True)

        # import kwplot
        # kwplot.autompl()
        # kwplot.plt.ion()
        # kwplot.imshow(data1, pnum=(1, 2, 1))
        # kwplot.imshow(data2, pnum=(1, 2, 2))


def test_off_by_one_with_small_img():
    """
    This originally did not work correctly because of warp was using
    integer-centers instead of integer-corners. The issue is described well in
    this article [WhereArePixels]_.

    References:
        .. [ResizeConfusion] https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
        .. [InvWarp] https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
        .. [TorchAffineTransform] https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
        .. [TorchIssue15386] https://github.com/pytorch/pytorch/issues/15386
        .. [WhereArePixels] https://ppwwyyxx.com/blog/2021/Where-are-Pixels/
        # https://github.com/pytorch/pytorch/issues/20785
        # https://github.com/pytorch/pytorch/pull/23923
        # https://github.com/pytorch/pytorch/pull/24929
        # https://user-images.githubusercontent.com/9757500/58150486-c5315900-7c34-11e9-9466-24f2bd431fa4.png

    SeeAlso:
        ~/code/kwimage/kwimage/util_warp.py
        ~/code/kwimage/kwimage/im_cv2.py

    """
    import delayed_image
    import kwimage
    import numpy as np
    raw = np.linspace(0, 1, 36).reshape(6, 6)
    delayed = delayed_image.DelayedIdentity(raw)

    warp = kwimage.Affine.coerce(offset=(-1e-8, -1e-9), scale=(8.6, 8.5))
    x = delayed.warp(warp)
    x.finalize(interpolation='nearest')

    warp = kwimage.Affine.coerce(offset=(0, 0), scale=(8.6, 8.5))
    x = delayed.warp(warp)
    data1 = x.finalize(interpolation='nearest')

    warp = kwimage.Affine.coerce(offset=(0, 0), scale=(2, 2))
    warp = warp @ kwimage.Affine.translate((0.5, 0.5))
    x = delayed.warp(warp)
    data2 = x.finalize(interpolation='nearest')

    data3 = kwimage.imresize(raw, scale=2.0, interpolation='nearest')

    SHOW = 0
    if SHOW:
        import kwplot
        pnum_ = kwplot.PlotNums(nRows=1, nCols=4)
        kwplot.autompl()
        kwplot.plt.ion()
        kwplot.imshow(raw, pnum=pnum_(), title='original image', show_ticks=True, origin_convention='corner')
        kwplot.imshow(kwimage.fill_nans_with_checkers(data1.copy()), pnum=pnum_(), title='scaled by non-integer', show_ticks=True, origin_convention='corner')
        kwplot.imshow(kwimage.fill_nans_with_checkers(data2.copy()), pnum=pnum_(), title='scaled by 2 and shifted by 0.5', show_ticks=True, origin_convention='corner')
        kwplot.imshow(kwimage.fill_nans_with_checkers(data3.copy()), pnum=pnum_(), title='imresize scale by 2', show_ticks=True, origin_convention='corner')

    raw.shape
    assert np.all(np.unique(raw) == np.unique(data1)), (
        'data1 should have exactly the same values as raw because it is '
        'just an upscale with nearest resampling. '
        'It should not have any nan values')

    assert not np.any(np.isnan(data2[1:, 1:])), (
        'data2 should not have any nan values except in the first row / column '
        'due to the 0.5 translation')

    assert np.all(np.isnan(data2[:1, :])), (
        'data2 first row should be all nans due to a shift by 0.5 and scale by 2')
    assert np.all(np.isnan(data2[:, :1])), (
        'data2 first row should be all nans due to a shift by 0.5 and scale by 2')

    assert not np.any(np.isnan(data3)), (
        'data3 is just a sanity check and should not have nans due to imresize implementation')

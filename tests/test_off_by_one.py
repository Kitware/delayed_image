
def test_off_by_one_with_upscale():
    import delayed_image
    import numpy as np
    delayed = delayed_image.DelayedLoad.demo(key='astro').prepare()

    orig = delayed.finalize()

    x = delayed.scale(2.0001)[0:4, 0:4]
    data1 = x.finalize(optimize=False, nodata_method='float')
    data2 = x.finalize(nodata_method='float')
    assert np.all(data2 == data1)

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
    orig = delayed.finalize()

    for s in np.linspace(0.4, 2.5, 100).tolist() + [0.5, 1.0, 2.0]:
        x = delayed.scale(s)[0:4, 0:4]
        data1 = x.finalize(optimize=False, nodata_method='float')
        data2 = x.finalize(nodata_method='float')
        assert np.all(data2 == data1)

        # import kwplot
        # kwplot.autompl()
        # kwplot.plt.ion()
        # kwplot.imshow(data1, pnum=(1, 2, 1))
        # kwplot.imshow(data2, pnum=(1, 2, 2))


def test_off_by_one_with_small_img():
    """
    This doesn't work right because of the align corners issue

    References:
        .. [ResizeConfusion] https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
        .. [InvWarp] https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
        .. [TorchAffineTransform] https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
        .. [TorchIssue15386] https://github.com/pytorch/pytorch/issues/15386
        # https://github.com/pytorch/pytorch/issues/20785
        # https://github.com/pytorch/pytorch/pull/23923
        # https://github.com/pytorch/pytorch/pull/24929
        # https://user-images.githubusercontent.com/9757500/58150486-c5315900-7c34-11e9-9466-24f2bd431fa4.png

    SeeAlso:
        ~/code/kwimage/kwimage/util_warp.py
        ~/code/kwimage/kwimage/im_cv2.py

    """
    import pytest
    pytest.skip('This is broken')

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
    data = x.finalize(interpolation='nearest')

    warp = kwimage.Affine.coerce(offset=(0, 0), scale=(2, 2))
    warp = warp @ kwimage.Affine.translate((0.5, 0.5))
    x = delayed.warp(warp)
    data2 = x.finalize(interpolation='nearest')

    data3 = kwimage.imresize(raw, scale=2.0, interpolation='nearest')

    if 1:
        import kwplot
        pnum_ = kwplot.PlotNums(nRows=1, nCols=4)
        kwplot.autompl()
        kwplot.plt.ion()
        kwplot.imshow(raw, pnum=pnum_())
        kwplot.imshow(kwimage.fill_nans_with_checkers(data), pnum=pnum_())
        kwplot.imshow(kwimage.fill_nans_with_checkers(data2), pnum=pnum_())
        kwplot.imshow(kwimage.fill_nans_with_checkers(data3), pnum=pnum_())

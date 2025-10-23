def test_crop_optimize_issue():
    """
    There was an issue in 0.2.0 where a crop would be optimized incorrectly.

    This is due to some interaction with overviews. When the data was not
    prepared, there wasn't any issue.

    The issue was that when you had
        * Warp(scale=2, dsize=(400, 400))
            * Dequantize(scale=2, dsize=(200, 200))
                * Overview(dsize=(200, 200))
                    * Load(dsize=(400, 400))

    It would optimize to
        * Dequantize(scale=2, dsize=(200, 200))
            * Load(dsize=(400, 400))

    Instead of
        * Dequantize(scale=2, dsize=(400, 400))
            * Load(dsize=(400, 400))
    """
    # from delayed_image import demo
    # demo.non_aligned_leafs()
    import delayed_image
    import numpy as np

    import pytest
    try:
        from osgeo import gdal  # NOQA
    except ImportError:
        pytest.skip()

    import ubelt as ub
    import kwimage
    dpath = ub.Path.appdir('delayed_image/tests/test_crop_optimize_issue')
    dpath.ensuredir()

    def demo_weird_delayed():
        # Simulate the weird issue in the saliency bands
        imdata = kwimage.grab_test_image()
        bband = kwimage.imresize(imdata[..., 2], dsize=(835, 549))
        sband = kwimage.imresize(imdata[..., 0], dsize=(582, 384))
        bband_fpath = kwimage.imwrite(dpath / 'dummy_blue.tif', bband, overviews=2)
        sband_fpath = kwimage.imwrite(dpath / 'dummy_salient.tif', sband, overviews=2)
        bandB = delayed_image.DelayedLoad(bband_fpath, dsize=(835, 549), channels='blue')
        bandS = delayed_image.DelayedLoad(sband_fpath, dsize=(582, 384), channels='salient')
        bandB = bandB.warp(np.eye(3) + 1e-8, dsize=(835, 549))
        bandS.meta['channels'] = delayed_image.FusedChannelSpec.coerce('salient')
        bandS = bandS.dequantize({'orig_max': 1})
        bandS = bandS.resize((835, 549))
        delayed = delayed_image.DelayedChannelConcat([bandB, bandS])
        delayed = delayed.resize((291, 192))
        delayed = delayed.resize((582, 384))
        return delayed

    delayed1 = demo_weird_delayed()
    delayed2 = demo_weird_delayed()

    chan1 = delayed1.take_channels('salient')
    chan2 = delayed2.take_channels('salient').prepare()

    print('\n-- Chan V1 [orig] --')
    chan1.write_network_text()
    print('\n-- Chan V2 [orig] --')
    chan2.write_network_text()

    print('\n-- Chan V1 [opt] --')
    chan1_opt = chan1.optimize()
    chan1_opt.write_network_text()
    print('\n-- Chan V2 [opt] --')
    chan2_opt = chan2.optimize()
    chan2_opt.write_network_text()

    assert chan1_opt.dsize == chan2_opt.dsize

    # chan1_opt._opt_logs
    # opt_logs1 = [d['obj']._opt_logs for n, d in chan1_opt._traversed_graph().nodes(data=True)]
    # opt_logs2 = [d['obj']._opt_logs for n, d in chan2_opt._traversed_graph().nodes(data=True)]
    # print(opt_logs1)
    # print(opt_logs2)


def test_lazy_warp_with_explicit_dsize():
    """
    In version 0.4.5 using lazy with warp would ignore the explicit dsize.
    Test that this is fixed. This error showed up in the geowatch msi example
    when doing kwcoco msi image generation.
    """
    import ubelt as ub
    import numpy as np
    import kwimage
    from delayed_image import DelayedLoad
    dpath = ub.Path.appdir('delayed_image/tests/lazy_warp_with_explicit_dsize')
    dpath.ensuredir()

    canvas_dsize = (431, 427)
    fpath1 = kwimage.imwrite(dpath / 'dummy1.tif', np.random.rand(72, 72))
    fpath2 = kwimage.imwrite(dpath / 'dummy2.tif', np.random.rand(427, 432))

    band1 = DelayedLoad(fpath1).prepare()
    band2 = DelayedLoad(fpath2).prepare()

    for lazy in [True, False]:
        # Test case with a real transform
        without_dsize1 = band1.warp({'scale': 6.0, 'type': 'affine'}, lazy=lazy)
        with_dsize1 = band1.warp({'scale': 6.0, 'type': 'affine'}, dsize=canvas_dsize, lazy=lazy)

        assert without_dsize1.dsize == (432, 432)
        assert with_dsize1.dsize == canvas_dsize

        # Test case with an identity transform
        without_dsize2 = band2.warp({'scale': 1.0, 'type': 'affine'}, lazy=lazy)
        with_dsize2 = band2.warp({'scale': 1.0, 'type': 'affine'}, dsize=canvas_dsize, lazy=lazy)
        assert without_dsize2.dsize != canvas_dsize
        assert without_dsize2.dsize == (432, 427)
        assert with_dsize2.dsize == canvas_dsize

        # Alternative, no transform is given
        without_dsize3 = band2.warp(None, lazy=lazy)
        with_dsize3 = band2.warp(None, dsize=canvas_dsize, lazy=lazy)
        assert without_dsize3.dsize != canvas_dsize
        assert without_dsize3.dsize == (432, 427)
        assert with_dsize3.dsize == canvas_dsize


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/delayed_image/tests/test_delayed_nodes.py
    """
    test_crop_optimize_issue()

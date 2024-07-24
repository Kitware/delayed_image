"""
Test the case where one image has a huge scale difference with respect to a
second image. For example, a UAV and Satellite image where there can be a 1000x
factor scale difference.
"""


def test_100x_scale_difference():
    """
    There is an issue here in that the native subdata does not seem to agree
    with the resampled subdata.

    References:
        https://forum.opencv.org/t/adapting-cv2-warpaffine-from-integer-corners-to-integer-centers-convention/15027/3
        https://ppwwyyxx.com/blog/2021/Where-are-Pixels/
    """
    import delayed_image
    import kwimage
    # import numpy as np

    def fancy_checkerboard(dsize, num_squares):
        checker = kwimage.checkerboard(dsize=dsize, num_squares=num_squares, on_value='red', off_value='yellow', bayer_value='gray')
        checker = kwimage.ensure_float01(checker)
        checker = kwimage.ensure_alpha_channel(checker, alpha=1.0)

        astro = kwimage.ensure_float01(kwimage.grab_test_image(dsize=dsize))
        astro = kwimage.ensure_float01(astro)
        astro = kwimage.ensure_alpha_channel(astro, alpha=0.3)

        raw = kwimage.overlay_alpha_images(astro, checker)
        raw = kwimage.ensure_float01(raw[..., 0:3], copy=True)
        return raw

    # Create an test image at a high resolution
    S1 = H1 = W1 = 1000
    raw1 = fancy_checkerboard((W1, H1), num_squares=100)
    delayed1 = delayed_image.DelayedIdentity(raw1)

    # Create an corresponding test image at a much lower
    S2 = H2 = W2 = 10
    raw2 = fancy_checkerboard((W2, H2), num_squares=10)
    delayed2 = delayed_image.DelayedIdentity(raw2)

    # Virtually upscale the low resolution and concatenate the aligned delayed
    # objects.
    delayed = delayed_image.DelayedChannelConcat([
        delayed1,
        delayed2.scale(
            S1 / S2,
            # border_value=float('nan'),
            # border_value='replicate'
        )
    ])
    delayed.print_graph()

    # Grab a small section of the data at the high virtual resolution
    box_size = 100

    region_of_interest = kwimage.Box.coerce([
        425, 225, box_size, box_size], format='xywh')

    # NOTE: this needs to be considered as coordinates, and not as indices
    region_of_interest = kwimage.Box.coerce([
        425, 120, box_size, box_size], format='xywh')

    # region_of_interest = kwimage.Box.coerce([
    #     -0.5, -0.5, box_size, box_size], format='xywh')

    slice_ = region_of_interest.to_slice()
    chip = delayed.crop(slice_, clip=False, wrap=False)
    chip.print_graph(fields='all')

    # Resampling Approach: Finalize to grab resampled data.
    print('Finalized resampled chip (nearest)')
    resampled_final = chip.finalize(interpolation='nearest', optimize=True)

    hires_resampled_sample = resampled_final[..., 0:3]
    lores_resampled_sample_nearest = resampled_final[..., 3:6]

    print('Finalized resampled chip (linear)')
    lores_resampled_sample_linear = chip.finalize(interpolation='linear', optimize=True)[..., 3:6]
    lores_resampled_sample_linear = kwimage.fill_nans_with_checkers(lores_resampled_sample_linear)

    # Native Approach:
    native_parts, native_warps = chip.optimize().undo_warps(remove=['scale'], return_warps=True)
    native1, native2 = native_parts
    warp_native1_from_virtual = native_warps[0]
    warp_native2_from_virtual = native_warps[1]

    # Use the warps to project the sample box onto the location in the original
    # image at the native resolution. This lets us shows which region we are
    # sampling in each image.
    roi_resolution1 = region_of_interest.warp(warp_native1_from_virtual)
    roi_resolution2 = region_of_interest.warp(warp_native2_from_virtual)

    print('Finalized native chip (hires)')
    hires_native_sample = native1.finalize()
    print('Finalized native chip (lores, nearest)')
    lores_native_sample_nearest = native2.finalize(interpolation='nearest')
    print('Finalized native chip (lores, linear)')
    lores_native_sample_linear = native2.finalize(interpolation='linear')

    # a = chip.optimize().parts[1].subdata.finalize()
    # a = chip.optimize().parts[1].finalize()
    # import kwplot
    # kwplot.imshow(a, fnum=4)

    # from rich import print
    print('\n[green]--- HiRes Box---')
    print(region_of_interest)
    print('\n[green]--- Native Box---')
    print(roi_resolution2)

    # Get the delayed operation tree for just the coarse image for print comparison
    resampled2 = chip.optimize().undo_warps(remove=[])[1]
    print('\n[green]--- Resampled Operations For Data 2 ---')
    resampled2.print_graph(fields='all')

    print('\n[green]--- Native Operations For Data 2 ---')
    native2.print_graph(fields='all')

    # Visual check to help ensure everything is ok
    import ubelt as ub
    DRAW = ub.argflag('--show')
    if DRAW:
        # Note the matplotlib grid (which has the center of the top left pixel
        # at 0,0) and the top left point is -0.5,-0.5, and the "delayed image
        # grid" might not agree
        import kwplot
        import functools
        kwplot.autompl()

        fig = kwplot.figure(fnum=1)
        fig.clf()

        num_rows = 3
        _imshow = functools.partial(kwplot.imshow, fnum=1, show_ticks=1, origin_convention='corner')

        # Draw where the box is on the native resolution datas
        _imshow(raw1, pnum=(num_rows, 3, 1), title='sample location in native resolution 1')
        # roi_resolution1.translate(-0.5).draw()
        # FIXME: need to handle drawing on matplotlib correctly
        roi_resolution1.draw()
        _imshow(raw2, pnum=(num_rows, 3, 2), title='sample location in native resolution 2')
        roi_resolution2.draw()
        # roi_resolution2.translate(-0.5).draw()

        # Show the resampled sample
        _imshow(hires_resampled_sample, pnum=(num_rows, 3, 4), title='resampled sample 1')
        _imshow(lores_resampled_sample_nearest, pnum=(num_rows, 3, 5), title='resampled sample 2 (nearest)')
        _imshow(lores_resampled_sample_linear, pnum=(num_rows, 3, 6), title='resampled sample 2 (linear)')

        # Show the native sample
        _imshow(hires_native_sample, pnum=(num_rows, 3, 7), title='native sample 1')
        _imshow(lores_native_sample_nearest, pnum=(num_rows, 3, 8), title='native sample 2 (nearest)')
        _imshow(lores_native_sample_linear, pnum=(num_rows, 3, 9), title='native sample 2 (linear)')

        # Test manual non-optimized resampled method
        # This seems to be a cv2 issue, I probably looked into this before
        kwplot.plt.show()

    # TODO: need to write checks beyond the visual one


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/delayed_image/tests/test_huge_scale_ratio.py --show
    """
    test_100x_scale_difference()

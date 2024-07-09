"""
Test the case where one image has a huge scale difference with respect to a
second image. For example, a UAV and Satellite image where there can be a 1000x
factor scale difference.
"""


def test_100x_scale_difference():
    import delayed_image
    # import numpy as np
    import kwimage

    # Create an test image at a high resolution
    S1 = H1 = W1 = 2000
    #raw1 = np.linspace(0, 1, H1 * W1).reshape(H1, W1)
    raw1 = kwimage.checkerboard(dsize=(W1, H1), num_squares=200)
    delayed1 = delayed_image.DelayedIdentity(raw1)

    # Create an corresponding test image at a much lower
    S2 = H2 = W2 = 20
    # raw2 = np.linspace(0, 1, H2 * W2).reshape(H2, W2)
    raw2 = kwimage.checkerboard(dsize=(W2, H2), num_squares=20)
    delayed2 = delayed_image.DelayedIdentity(raw2)

    # Virtually upscale the low resolution and concatenate the aligned delayed
    # objects.
    delayed = delayed_image.DelayedChannelConcat([
        delayed1,
        delayed2.scale(S1 / S2)
    ])
    delayed.print_graph()

    # Grab a small section of the data at the high virtual resolution
    box_size = 100

    region_of_interest = kwimage.Box.coerce([
        1025, 525, box_size, box_size], format='xywh')
    slice_ = region_of_interest.to_slice()
    chip = delayed[slice_]
    chip = chip.optimize()
    chip.print_graph()

    # Resampling Approach: Finalize to grab resampled data.
    resampled_final = chip.finalize(interpolation='nearest')

    hires_resampled_sample = resampled_final[..., 0]
    lores_resampled_sample = resampled_final[..., 1]

    # Native Approach:
    native_parts, native_warps = chip.undo_warps(remove=['scale'], return_warps=True)
    native1, native2 = native_parts
    warp_native1_from_virtual = native_warps[0]
    warp_native2_from_virtual = native_warps[1]

    # Use the warps to project the sample box onto the location in the original
    # image at the native resolution. This lets us shows which region we are
    # sampling in each image.
    roi_resolution1 = region_of_interest.warp(warp_native1_from_virtual)
    roi_resolution2 = region_of_interest.warp(warp_native2_from_virtual)

    hires_native_sample = native1.finalize()
    lores_native_sample = native2.finalize()

    native2.print_graph()

    # Visual check to help ensure everything is ok
    DRAW = 1
    if DRAW:
        import kwplot
        kwplot.autompl()

        kwplot.imshow(raw1, pnum=(4, 2, 1), fnum=1, title='native resolution 1')
        kwplot.imshow(raw2, pnum=(4, 2, 2), fnum=1, title='native resolution 2')

        # Draw where the box is on the native resolution datas
        kwplot.imshow(raw1, pnum=(4, 2, 3), fnum=1, title='sample location in native resolution 1')
        roi_resolution1.draw()
        kwplot.imshow(raw2, pnum=(4, 2, 4), fnum=1, title='sample location in native resolution 2')
        roi_resolution2.draw()

        # Show the resampled sample
        kwplot.imshow(hires_resampled_sample, pnum=(4, 2, 5), fnum=1, title='resampled resolution sample 1')
        kwplot.imshow(lores_resampled_sample, pnum=(4, 2, 6), fnum=1, title='resampled resolution sample 2')

        # Show the native sample
        kwplot.imshow(hires_native_sample, pnum=(4, 2, 7), fnum=1, title='native resolution sample 1')
        kwplot.imshow(lores_native_sample, pnum=(4, 2, 8), fnum=1, title='native resolution sample 2')

    # TODO: need to write checks beyond the visual one

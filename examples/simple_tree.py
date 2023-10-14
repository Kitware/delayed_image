def demo_simple_tree():
    """
    This shows a simple tree of image operations
    """
    from delayed_image import DelayedChannelConcat
    import delayed_image
    import kwimage
    import numpy as np
    import ubelt as ub

    r = delayed_image.DelayedLoad.demo(channels='rgb', overviews=3, dsize=(256, 256)).take_channels([0])
    g = delayed_image.DelayedLoad.demo(channels='rgb', overviews=3, dsize=(128, 128)).take_channels([1])
    b = delayed_image.DelayedLoad.demo(channels='rgb', overviews=3, dsize=(512, 512)).take_channels([2])

    g = g.scale(2.0)
    b = b.scale(0.5)

    rgb = DelayedChannelConcat([r, g, b])

    transform = kwimage.Affine.affine(theta=np.pi / 8, about=(128, 128))
    rotated = rgb.warp(transform)
    scaled = rotated.scale(0.5)
    cropped = scaled[10:120, 20:133]

    rescaled = cropped.scale(3.0)
    naive = rescaled.dequantize({'orig_max': 1.0, 'orig_min': 0.0, 'quant_max': 255})

    naive.print_graph()

    optimized = naive.optimize()

    optimized.print_graph()

    with ub.Timer(label='naive finalize') as t1:
        im_naive = naive.finalize(optimize=False)

    with ub.Timer(label='optimized finalize') as t2:
        im_optimized = optimized.finalize(optimize=False)

    im_diff = np.abs(im_naive - im_optimized)

    im_diff = kwimage.fill_nans_with_checkers(im_diff)
    im_optimized = kwimage.fill_nans_with_checkers(im_optimized)
    im_naive = kwimage.fill_nans_with_checkers(im_naive)

    import kwplot
    kwplot.autompl()
    naive_canvas = kwimage.draw_header_text(im_optimized, f'naive\n{t1.elapsed:0.4f}s', fit='shrink')
    opt_canvas = kwimage.draw_header_text(im_optimized, f'optimized\n{t2.elapsed:0.4f}s', fit='shrink')
    diff_canvas = kwimage.draw_header_text(im_diff, 'difference\nimage', fit='shrink')
    stacked = kwimage.stack_images([opt_canvas, naive_canvas, diff_canvas], axis=1, pad=10, bg_value='darkgreen')
    kwplot.imshow(stacked)

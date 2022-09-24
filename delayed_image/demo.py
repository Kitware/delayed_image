def non_aligned_leafs():
    from delayed_image import DelayedLoad
    from delayed_image import DelayedChannelConcat
    import numpy as np

    base = DelayedLoad.demo(channels='r|g|b').prepare().dequantize({'quant_max': 255})
    bandR = base[:, :, 0].scale(100 / 512)[:, :-50].evaluate()
    bandG = base[:, :, 1].scale(300 / 512).warp({'theta': np.pi / 8, 'about': (150, 150)}).evaluate()
    bandB = base[:, :, 2].scale(600 / 512)[:150, :].evaluate()
    # Align the bands in "video" space
    delayed_vidspace = DelayedChannelConcat([
        bandR.scale(6, dsize=(600, 600)).optimize(),
        bandG.warp({'theta': -np.pi / 8, 'about': (150, 150)}).scale(2, dsize=(600, 600)).optimize(),
        bandB.scale(1, dsize=(600, 600)).optimize(),
    ]).warp(
        #{'scale': 0.35, 'theta': 0.3, 'about': (30, 50), 'offset': (-10, -80)}
        {'scale': 0.7}
    )
    return delayed_vidspace

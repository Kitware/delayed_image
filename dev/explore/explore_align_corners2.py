"""
Explores the align corners behavior of various kwimage functions and their
underlying cv2 / torch implementations.

TODO:

Handle different settings of align corners in both imresize and warp_affine.
Behaviors should follow:


    +---------------+-----------------------+
    | align_corners | pixels interpretation |
    +---------------+-----------------------+
    | True          | points in a grid      |
    +---------------+-----------------------+
    | False         | areas of 1x1 squares  |
    +---------------+-----------------------+

References:
    https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
    https://medium.com/@elagwoog/you-might-have-misundertood-the-meaning-of-align-corners-c681d0e38300
    https://user-images.githubusercontent.com/9757500/58150486-c5315900-7c34-11e9-9466-24f2bd431fa4.png
    https://forum.opencv.org/t/adapting-cv2-warpaffine-from-integer-corners-to-integer-centers-convention/15027
    https://ppwwyyxx.com/blog/2021/Where-are-Pixels/


SeeAlso:
    Notes in warp_tensor

"""
import numpy as np
import kwimage


def main():
    D1 = 2
    D2 = 12

    raw_scale = D2 / D1
    new_dsize = (D2, 1)

    results = {}

    img = np.arange(D1)[None, :].astype(np.float32)
    img_imresize = kwimage.imresize(img, dsize=new_dsize, interpolation='linear')
    results['imresize'] = img_imresize

    # Shift and scale (align_corners=True)
    T1 = kwimage.Affine.translate((+0.5, 0))
    T2 = kwimage.Affine.translate((-0.5, 0))
    S = kwimage.Affine.scale((raw_scale, 1))
    S2 = T2 @ S @ T1
    img_shiftscale = kwimage.warp_affine(img, S2, dsize=new_dsize, border_mode='reflect')
    results['aff_shiftscale'] = img_shiftscale

    import torch  # NOQA
    input = torch.from_numpy(img)[None, :, : , None]
    results['torch_ac0'] = torch.nn.functional.interpolate(input, size=new_dsize, mode='bilinear', align_corners=False)[0, :, :, 0].numpy()

    # Pure scaling (align_corners=True)
    S = kwimage.Affine.scale((raw_scale, 1))
    img_rawscale = kwimage.warp_affine(img, S, dsize=new_dsize)
    results['aff_scale'] = img_rawscale
    results['torch_ac1'] = torch.nn.functional.interpolate(input, size=new_dsize, mode='bilinear', align_corners=True)[0, :, :, 0].numpy()

    import pandas as pd
    import ubelt as ub
    import rich
    df = pd.DataFrame(ub.udict(results).map_values(lambda x: x.ravel()))
    rich.print(df.to_string())


def new_coordinate_understanding():
    """
    To properly deal with slicing and warping images we need to know:

        1. what is the coordinate system?
        2. what is the indexing system?
        3. what is origin of the coordinate system?

    In other words, an image is a discrete grid of samples that corresponds to
    some underlying coordinate system. We need to know the coordinates that
    corespond to the centers and edges of bins in the sample grid.

    When working with slices, you often work in the array indexing system, and
    we should be careful when mixing this with affine warps which work on the
    underling coordinate system.

    * The indexing system is simply addresses into the array that stores
    discrete samples.

    * The coordinate system corresponds to an underlying continuous signal that
    is being sampled.

    We need an API that is clear and explicit about which ones the users want.

    We always want align_corners=False


    Use case:

        we have a box

        kwimage.Box.coerce([0, 0, 10, 10], format='xywh')

        does the box represent coordinates in the underlying signal or indicies
        in the image?

        If we are using integer-centers, then
        the slice [0:10, 0:10]
        corresponds to coordinates [-0.5, 9.5], [-0.5, 9.5]
        these are topologically closed intervals, but should they be open?
        is this a valid option? if intervals are closed or open?

        If we are using integer-corners then the slice [0:10, 0:10]
        corresponds to coordinates [0.0, 10.0], [0.0, 10.0]
    """
    import kwplot
    import kwimage
    import numpy as np

    kwplot.autompl()

    xx, yy = np.meshgrid(np.arange(8), np.arange(8))
    points = kwimage.Points(xy=np.stack([xx.ravel(), yy.ravel()]).T)

    # Create an checkerboard
    H2 = W2 = 8
    raster = kwimage.checkerboard(dsize=(W2, H2), num_squares=8)
    colors = np.zeros((H2, W2, 3))
    colors[0::2, 0::2, :] = np.array(kwimage.Color.coerce('kitware_red').as01())[None, None, :]
    colors[2::4, 0::2, :] = np.array(kwimage.Color.coerce('kitware_yellow').as01())[None, None, :]
    colors[1::2, 1::2, :] = np.array(kwimage.Color.coerce('kitware_green').as01())[None, None, :]
    raster = kwimage.atleast_3channels(raster) * colors.round(1)

    transform = kwimage.Affine.scale(2)
    upscaled_raster = kwimage.warp_affine(raster, transform, dsize='auto', interpolation='nearest')
    upscaled_points = points.warp(transform)

    offset1 = kwimage.Affine.coerce(offset=.5)
    offset2 = kwimage.Affine.coerce(offset=-.5)
    modified_transform = offset2 @ transform @ offset1

    modified_upscaled_raster = kwimage.warp_affine(raster, modified_transform, dsize='auto', interpolation='nearest')

    kwplot.figure(fnum=1, doclf=1, pnum=(2, 2, 1))
    kwplot.imshow(raster, show_ticks=True, title='integer-centers, original')
    points.draw(radius=0.2, color='kitware_blue')

    kwplot.figure(fnum=1, pnum=(2, 2, 2))
    kwplot.imshow(upscaled_raster, show_ticks=True, title='integer-centers, upscaled')
    upscaled_points.draw(radius=0.5, color='kitware_blue')

    kwplot.figure(fnum=1, pnum=(2, 2, 3))
    kwplot.imshow(raster, show_ticks=True, pixels_are='points', title='integer-corners, original')
    points.draw(radius=0.2, color='kitware_blue')

    kwplot.figure(fnum=1, pnum=(2, 2, 4))
    kwplot.imshow(modified_upscaled_raster, show_ticks=True, pixels_are='points', title='integer-corners, upscaled')
    upscaled_points.draw(radius=0.5, color='kitware_blue')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/delayed_image/dev/explore/explore_align_corners2.py
    """
    main()

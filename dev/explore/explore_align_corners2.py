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

    # Define the transform of interest
    transform = kwimage.Affine.scale(3)

    # modified_transform = transform @ offset1
    # transform = modified_transform

    # Define the original raster size
    H1 = W1 = 4

    if 1:
        # simple auto dsize that probably needs to be fixed
        box = kwimage.Boxes(np.array([[0, 0, W1, H1]]), 'xywh')
        warped_box = box.warp(transform)
        new_dsize = tuple(map(int, warped_box.to_ltrb().quantize().data[0, 2:4]))
        W2, H2 = new_dsize

    # Define the indexes of the pixels
    x_pixel_index_basis, y_pixel_index_basis = np.meshgrid(np.arange(H1), np.arange(W1))

    # Define the coordinates of the centers of the pixels (using the integer-center convention)
    pixel_int_center_coords = kwimage.Points(xy=np.stack([
        x_pixel_index_basis.ravel(),
        y_pixel_index_basis.ravel()
    ]).T.astype(float))

    pixel_int_corner_coords = pixel_int_center_coords.translate(0.5)

    def grid_basis_and_segments(W, H, corner=0.5):
        xmin, ymin = corner, corner
        xmax = W - corner
        ymax = H - corner
        x_basis = np.linspace(xmin, xmax, num=W + 1)
        y_basis = np.linspace(ymin, ymax, num=H + 1)
        # Define endpoints for line segements that define the pixel edges.
        edge_segments = []
        for x in x_basis:
            edge_segments.append([[x, ymin], [x, ymax]])
        for y in y_basis:
            edge_segments.append([[xmin, y], [xmax, y]])
        edge_segments = np.array(edge_segments)
        return x_basis, y_basis, edge_segments

    # Define endpoints for line segements that define the pixel edges.
    x_pixel_edge_coord_basis, y_pixel_edge_coord_basis, pixel_edge_segments = grid_basis_and_segments(W1, H1, corner=0)

    # Get the grid for the resampled pixel edges
    x_resampled_pixel_edge_coord_basis, y_resampled_pixel_edge_coord_basis, resampled_pixel_edge_segments = grid_basis_and_segments(W2, H2, corner=0)

    warped_pixel_edge_segments = kwimage.warp_points(transform, pixel_edge_segments)
    inv_resampled_pixel_edge_segments = kwimage.warp_points(transform.inv(), resampled_pixel_edge_segments)

    # Create the raster checkerboard
    raster = kwimage.checkerboard(dsize=(W1, H1), num_squares=W1,
                                  on_value='kitware_green',
                                  off_value='kitware_red',
                                  bayer_value='kitware_yellow')

    warped_rasterN = kwimage.warp_affine(raster, transform, dsize=(W2, H2),
                                         interpolation='linear', origin_convention='center')
    warped_rasterL = kwimage.warp_affine(raster, transform, dsize=(W2, H2),
                                         interpolation='nearest', origin_convention='center')

    warped_pixel_int_center_coords = pixel_int_center_coords.warp(transform)
    warped_pixel_int_corner_coords = pixel_int_corner_coords.warp(transform)

    modified_warped_rasterN = kwimage.warp_affine(
        raster, transform, dsize=(W2, H2), interpolation='nearest', origin_convention='corner')
    modified_warped_rasterL = kwimage.warp_affine(
        raster, transform, dsize=(W2, H2), interpolation='linear', origin_convention='corner')

    kwplot.figure(fnum=1, doclf=1, pnum=(2, 3, 1))
    kwplot.imshow(raster, show_ticks=True, title='integer-centers, original')
    pixel_int_center_coords.draw(radius=0.1, color='kitware_darkblue')

    kwplot.figure(fnum=1, pnum=(2, 3, 2))
    kwplot.imshow(warped_rasterL, show_ticks=True, title='integer-centers, warped (nearest)')
    warped_pixel_int_center_coords.draw(radius=0.2, color='kitware_darkblue')

    kwplot.figure(fnum=1, pnum=(2, 3, 3))
    kwplot.imshow(warped_rasterN, show_ticks=True, title='integer-centers, warped (linear)')
    warped_pixel_int_center_coords.draw(radius=0.2, color='kitware_darkblue')

    ax = kwplot.figure(fnum=1, pnum=(2, 3, 4)).gca()
    kwplot.imshow(raster, show_ticks=True, origin_convention='corner', title='integer-corners, original')
    pixel_int_corner_coords.draw(radius=0.1, color='kitware_darkblue')
    artman = kwplot.ArtistManager()
    for segment in pixel_edge_segments:
        artman.add_linestring(segment, color='kitware_darkblue', linewidth=4)
    for segment in inv_resampled_pixel_edge_segments:
        artman.add_linestring(segment, color='kitware_blue', linewidth=1)
    artman.add_to_axes(ax)

    ax = kwplot.figure(fnum=1, pnum=(2, 3, 5)).gca()
    kwplot.imshow(modified_warped_rasterN, show_ticks=True, origin_convention='corner', title='integer-corners, warped (nearest)')
    warped_pixel_int_corner_coords.draw(radius=0.2, color='kitware_darkblue')
    artman = kwplot.ArtistManager()
    for segment in warped_pixel_edge_segments:
        artman.add_linestring(segment, color='kitware_darkblue', linewidth=4)
    for segment in resampled_pixel_edge_segments:
        artman.add_linestring(segment, color='kitware_blue', linewidth=1)
    artman.add_to_axes(ax)

    ax = kwplot.figure(fnum=1, pnum=(2, 3, 6)).gca()
    kwplot.imshow(modified_warped_rasterL, show_ticks=True, origin_convention='corner', title='integer-corners, warped (linear)')
    warped_pixel_int_corner_coords.draw(radius=0.2, color='kitware_darkblue')
    artman = kwplot.ArtistManager()
    for segment in warped_pixel_edge_segments:
        artman.add_linestring(segment, color='kitware_darkblue', linewidth=4)
    for segment in resampled_pixel_edge_segments:
        artman.add_linestring(segment, color='kitware_blue', linewidth=1)
    artman.add_to_axes(ax)

    # import kwplot
    # fig = kwplot.figure(fnum=2, pnum=(1, 2, 1), doclf=1)
    # ax = fig.gca()
    # ax.set_xlim(-2, W1 + 2.0)
    # ax.set_ylim(-2, H1 + 2.0)
    # kwplot.imshow(raster, pixels_are='points', show_ticks=True)
    # artman = kwplot.ArtistManager()
    # for segment in pixel_edge_segments:
    #     artman.add_linestring(segment, color='kitware_darkblue', linewidth=4)
    # for segment in inv_resampled_pixel_edge_segments:
    #     artman.add_linestring(segment, color='kitware_blue', linewidth=1)
    # artman.add_to_axes(ax)

    # ax = kwplot.figure(fnum=2, pnum=(1, 2, 2)).gca()
    # ax.set_xlim(-2, W2 + 2.0)
    # ax.set_ylim(-2, H2 + 2.0)
    # # ax.set_xlim(xmin - 0.5, xmax + 0.5)
    # # ax.set_ylim(ymin - 0.5, ymax + 0.5)
    # artman = kwplot.ArtistManager()
    # kwplot.imshow(modified_warped_raster, pixels_are='points', show_ticks=True)
    # for segment in warped_pixel_edge_segments:
    #     artman.add_linestring(segment, color='kitware_darkblue', linewidth=4)
    # for segment in resampled_pixel_edge_segments:
    #     artman.add_linestring(segment, color='kitware_blue', linewidth=1)
    # artman.add_to_axes(ax)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/delayed_image/dev/explore/explore_align_corners2.py
    """
    main()

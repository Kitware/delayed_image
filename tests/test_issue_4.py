def test_issue4():
    """
    The symptom is given this tree:

        ╙── Warp dsize=(225,225),transform={offset=(0.0000,21.0000)}
            ╽
            Crop dsize=(0,204.0000),space_slice=(slice(0,204.0,None),slice(219,219,None))
            ╽
            Warp dsize=(219,219),transform={offset=(-0.0000,0.0000),scale=(0.9992,0.9992)}
            ╽
            ChannelConcat axis=2
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=blue,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_blue.tif
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=green,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_green.tif
            └─╼ Warp dsize=(219,219),transform={}
                ╽
                Load channels=red,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_red.tif

    Optimize resulted in:

        ~/.pyenv/versions/3.11.2/lib/python3.11/site-packages/delayed_image/helpers.py in _swap_warp_after_crop(root_region_bounds, tf_leaf_to_root)
            157
            158         # TODO: test the case where old_w or old_h are zero
        --> 159         padw = int(np.ceil(leaf_w / old_w))
            160         padh = int(np.ceil(leaf_h / old_h))
            161     else:

        OverflowError: cannot convert float infinity to integer



    The sequence of operations to build the tree was:

            delayed_frame = coco_img.imdelay(
                channels=request_chanspec, space=space,
                interpolation=interpolation,
                nodata_method=nodata,
                antialias=antialias
            )

        In [175]: delayed_frame.write_network_text()
        ╙── Warp dsize=(219,219),transform={offset=(-0.0000,0.0000),scale=(0.9992,0.9992)}
            ╽
            ChannelConcat axis=2
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=blue,dsize=(219,219),fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_blue.tif
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=green,dsize=(219,219),fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_green.tif
            └─╼ Warp dsize=(219,219),transform={}
                ╽
                Load channels=red,dsize=(219,219),fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_red.tif


        In [178]: requested_space_slice
        Out[178]: (slice(-21.0, 204.0, None), slice(219, 444.0, None))

        In [179]: space_pad
        Out[179]: [(0, 0), (0, 0)]

        delayed_crop = delayed_frame.crop(requested_space_slice,
                                          clip=False, wrap=False,
                                          pad=space_pad)
        delayed_crop = delayed_crop.prepare()

        In [177]: delayed_crop.write_network_text()
        ╙── Warp dsize=(225,225),transform={offset=(0.0000,21.0000)}
            ╽
            Crop dsize=(0,204.0000),space_slice=(slice(0,204.0,None),slice(219,219,None))
            ╽
            Warp dsize=(219,219),transform={offset=(-0.0000,0.0000),scale=(0.9992,0.9992)}
            ╽
            ChannelConcat axis=2
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=blue,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_blue.tif
            ├─╼ Warp dsize=(219,219),transform={}
            │   ╽
            │   Load channels=green,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_green.tif
            └─╼ Warp dsize=(219,219),transform={}
                ╽
                Load channels=red,dsize=(219,219),num_overviews=6,fname=crop_20191209T150000Z_N04.596732W074.222119_N04.600692W074.218175_WV_0_red.tif

        delayed_crop = delayed_crop.optimize()

        Error

    Note:
        The issue seems to be that the crop is ouside the bounds of the image,
        so we just need to ensure the divide by zero doesn't happen.

    """
    from delayed_image import DelayedChannelConcat
    import delayed_image
    import kwimage
    import numpy as np

    r = delayed_image.DelayedLoad.demo(channels='r', overviews=6, dsize=(219, 219))
    g = delayed_image.DelayedLoad.demo(channels='g', overviews=6, dsize=(219, 219))
    b = delayed_image.DelayedLoad.demo(channels='b', overviews=6, dsize=(219, 219))

    concat = DelayedChannelConcat([r.warp({}), g.warp({}), b.warp({})])

    mat = kwimage.Affine(np.array([[ 9.99169700e-01,  0.00000000e+00, -5.82076609e-11],
                                   [ 0.00000000e+00,  9.99169700e-01,  2.91038305e-11],
                                   [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]))
    delayed_frame = concat.warp(mat)
    delayed_frame.write_network_text()

    requested_space_slice = (slice(-21.0, 204.0, None), slice(219, 444.0, None))
    space_pad = [(0, 0), (0, 0)]
    delayed_crop = delayed_frame.crop(requested_space_slice,
                                      clip=False, wrap=False,
                                      pad=space_pad)
    delayed_crop = delayed_crop.prepare()

    delayed_crop.write_network_text()
    optimized = delayed_crop.optimize()
    optimized.write_network_text()
    assert optimized.dsize == (225, 225)

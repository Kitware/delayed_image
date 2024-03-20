Delayed Image
=============

|GitlabCIPipeline| |GitlabCICoverage| |Pypi| |PypiDownloads| |ReadTheDocs|


+------------------+-------------------------------------------------------------+
| Read the docs    | https://delayed-image.readthedocs.io                        |
+------------------+-------------------------------------------------------------+
| Gitlab (main)    | https://gitlab.kitware.com/computer-vision/delayed_image    |
+------------------+-------------------------------------------------------------+
| Github (mirror)  | https://github.com/Kitware/delayed_image                    |
+------------------+-------------------------------------------------------------+
| Pypi             | https://pypi.org/project/delayed-image                      |
+------------------+-------------------------------------------------------------+

The delayed image module lets you describe (a tree of) image operations, but
waits to execute them until the user calls ``finalize``. This allows for a
sequence of operations to be optimized before it is executed, which means
faster execution and fewer quantization artifacts.

The optimization logic can also leverage image formats that contain builtin
tiling or overviews using the GDAL library. Formats that contain tiles allow
delayed image to read only a subset of the image, if only a small part is
cropped to it.  Overviews allow delayed image to load pre-downscaled images if
it detects a scaling operation. This is **much** faster than the naive way of
accomplishing these operations, and **much** easier than having to remember to
do everything in the right order yourself.

Note: GDAL is optional, but recommended. Precompiled GDAL wheels are available
on Kitware's `large image wheel repository <https://girder.github.io/large_image_wheels/>`__.
Use ``pip install gdal -f https://girder.github.io/large_image_wheels/``
to install GDAL from this server. Track status of official GDAL wheels `here
<https://github.com/OSGeo/gdal/issues/3060>`__.


History
-------

This module is still in its early days of development and is a port the
following code from kwcoco:

* ChannelSpec
* SensorChannelSpec
* DelayedLoad & other delayed operations


Quick Start
-----------

.. code:: python

    # Given a path to some image
    import kwimage
    fpath = kwimage.grab_test_image_fpath('amazon')

    # Demo how to load, scale, and crop a part of an image.
    import delayed_image
    delayed = delayed_image.DelayedLoad(fpath)
    delayed = delayed.prepare()
    delayed = delayed.scale(0.1)
    delayed = delayed[128:256, 128:256]

    import kwplot
    kwplot.autompl()
    kwplot.imshow(delayed.finalize())
    kwimage.imwrite('foo.png', delayed.finalize())

.. image:: https://i.imgur.com/lsWLkPx.png

See `the quickstart jupyter notebook <examples/quickstart.ipynb/>`__ for more details.

Delayed Loading
---------------

Example of delayed loading:

.. code:: python

    >>> from delayed_image import DelayedLoad
    >>> import kwimage
    >>> fpath = kwimage.grab_test_image_fpath(overviews=3)
    >>> dimg = DelayedLoad(fpath, channels='r|g|b').prepare()
    >>> quantization = {'quant_max': 255, 'nodata': 0}
    >>> #
    >>> # Make a complex chain of operations
    >>> dimg = dimg.dequantize(quantization)
    >>> dimg = dimg.warp({'scale': 1.1})
    >>> dimg = dimg.warp({'scale': 1.1})
    >>> dimg = dimg[0:400, 1:400]
    >>> dimg = dimg.warp({'scale': 0.5})
    >>> dimg = dimg[0:800, 1:800]
    >>> dimg = dimg.warp({'scale': 0.5})
    >>> dimg = dimg[0:800, 1:800]
    >>> dimg = dimg.warp({'scale': 0.5})
    >>> dimg = dimg.warp({'scale': 1.1})
    >>> dimg = dimg.warp({'scale': 1.1})
    >>> dimg = dimg.warp({'scale': 2.1})
    >>> dimg = dimg[0:200, 1:200]
    >>> dimg = dimg[1:200, 2:200]
    >>> dimg.write_network_text()
    ╙── Crop dsize=(128,130),space_slice=(slice(1,131,None),slice(2,130,None))
        ╽
        Crop dsize=(130,131),space_slice=(slice(0,131,None),slice(1,131,None))
        ╽
        Warp dsize=(131,131),transform={scale=2.1000}
        ╽
        Warp dsize=(62,62),transform={scale=1.1000}
        ╽
        Warp dsize=(56,56),transform={scale=1.1000}
        ╽
        Warp dsize=(50,50),transform={scale=0.5000}
        ╽
        Crop dsize=(99,100),space_slice=(slice(0,100,None),slice(1,100,None))
        ╽
        Warp dsize=(100,100),transform={scale=0.5000}
        ╽
        Crop dsize=(199,200),space_slice=(slice(0,200,None),slice(1,200,None))
        ╽
        Warp dsize=(200,200),transform={scale=0.5000}
        ╽
        Crop dsize=(399,400),space_slice=(slice(0,400,None),slice(1,400,None))
        ╽
        Warp dsize=(621,621),transform={scale=1.1000}
        ╽
        Warp dsize=(564,564),transform={scale=1.1000}
        ╽
        Dequantize dsize=(512,512),quantization={quant_max=255,nodata=0}
        ╽
        Load channels=r|g|b,dsize=(512,512),num_overviews=3,fname=astro_overviews=3.tif

    >>> # Optimize the chain
    >>> dopt = dimg.optimize()
    >>> dopt.write_network_text()
    ╙── Warp dsize=(128,130),transform={offset=(-0.6115,-1.0000),scale=1.5373}
        ╽
        Dequantize dsize=(80,83),quantization={quant_max=255,nodata=0}
        ╽
        Crop dsize=(80,83),space_slice=(slice(0,83,None),slice(3,83,None))
        ╽
        Overview dsize=(128,128),overview=2
        ╽
        Load channels=r|g|b,dsize=(512,512),num_overviews=3,fname=astro_overviews=3.tif

    #
    >>> final0 = dimg.finalize(optimize=False)
    >>> final1 = dopt.finalize()
    >>> assert final0.shape == final1.shape
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.imshow(final0, pnum=(1, 2, 1), fnum=1, title='raw')
    >>> kwplot.imshow(final1, pnum=(1, 2, 2), fnum=1, title='optimized')


.. image:: https://i.imgur.com/3SGvxtC.png


Native Resolution Sampling
--------------------------

Consider the case where we have multiple images on disk in different
resolutions, but they correspond to the same scene (e.g. a satellite image may
have RGB bands at 10 meter resolution and an infrared band at 30 meter
resolution), and we want to sample corresponding regions in each image.
Typically a developer may opt to simply rescale everything to the same
resolution, so everything corresponds and then just crop out the region.  This
works but it has the negative effect of incurring resampling artifacts.

Delayed image allows for easy and intuitive "native resolution sampling".  We
can perform a delayed scale operation to get a "view" of an image as if we
rescaled all component bands to the same resolution, and then perform a delayed
crop. Finalizing this delayed operation is exactly the same as the previously
described case (except that it benefits from delayed image's optimized
operation reordering). However, we can go further. Because we know about the
underlying operation graph we can undo the scale component while keeping the
crop component, which results in loading the corresponding parts of the image
inside the cropped area, but does not do any resampling. The images on disk can
differ in more than just resolution, they could also be offset, skewed or
rotated, and this unwarping procedure will still work.

The following image illustrates an extreme example of this were we simulate a
low resolution red band (R), a medium but rotated resolution green band (G),
and a high but cropped resolution blue (B) band.

.. image:: https://i.imgur.com/fW7Mdo1.png


The raw bands on disk are shown in the top row. The second row demonstrates the
aligned space that we can conceptually think in when performing the crop. The
blue box defined in this row and is projected to all other images using delayed
image. The third row shows the result of the naive resampled alignment and
cropping of the blue box (and also pixel differences between optimized and
non-optimized finalizations). Lastly the fourth row shows the native sampling where
each crop corresponds to the same region, but we have removed all scale factors
(rotation and skew resamplings are still done to align to image corners up to a
scale factor).

For code details see the doctest in `delayed_image/__init__.py __doc__:2 <https://gitlab.kitware.com/computer-vision/delayed_image/-/blob/main/delayed_image/__init__.py#L115>`_


SensorChanSpec
--------------

Includes the SensorChan spec, which makes handling channels from different
sensing sources easier.

The sensor/channel spec isn't necessary to use delayed image, but it helps ---
particularly the channel spec --- to be able to semantically label the channels
when performing delayed load operations.

On a simple level all you need to know to use the basic channel spec is that
channel names are ``|`` delimited. E.g. ``red|green|blue`` refers to a 3
channel image. You can use these names to select subsets of channels. Here is
an example where you load an image, provide it with the semantic labels for
each channel, and then use them to select a single channel.

.. code:: python

    import delayed_image
    import kwimage
    fpath = kwimage.grab_test_image_fpath(overviews=3)

    # When you create a delayed image, you can enrich the image with
    # information about what channels it contains by specifying the
    # channels attribute.
    delayed = DelayedLoad(fpath, channels='red|green|blue').prepare()

    # You can use this to semantically interact with the channels
    delayed_g = delayed.take_channels('green')
    assert delayed_g.shape == (512, 512, 1)

    # Specifying more than one channel works too
    delayed_rb = delayed.take_channels('blue|red')
    assert delayed_rb.shape == (512, 512, 2)


Much of the Sensor/Channel spec functionality exists for the benefit of other
projects like `kwcoco <https://gitlab.kitware.com/computer-vision/kwcoco>`_.
Admittedly, this library isn't the perfect home for the full sensor / channel
spec, but this is where it currently lives.

The full sensor channel spec has a formal grammar defined in this package.

 .. code::

    // SENSOR_CHAN_GRAMMAR
    ?start: stream

    // An identifier can contain spaces
    IDEN: ("_"|"*"|LETTER) ("_"|" "|"-"|"*"|LETTER|DIGIT)*

    chan_single : IDEN
    chan_getitem : IDEN "." INT
    chan_getslice_0b : IDEN ":" INT
    chan_getslice_ab : (IDEN "." INT ":" INT) | (IDEN ":" INT ":" INT)

    // A channel code can just be an ID, or it can have a getitem
    // style syntax with a scalar or slice as an argument
    chan_code : chan_single | chan_getslice_0b | chan_getslice_ab | chan_getitem

    // Fused channels are an ordered sequence of channel codes (without sensors)
    fused : chan_code ("|" chan_code)*

    // A channel only part can be a fused channel or a sequence
    channel_rhs : fused | fused_seq

    // Channels can be specified in a sequence but must contain parens
    fused_seq : "(" fused ("," fused)* ")"

    // Sensors can be specified in a sequence but must contain parens
    sensor_seq : "(" IDEN ("," IDEN)* "):"

    sensor_lhs : (IDEN ":") | (sensor_seq)

    sensor_chan : sensor_lhs channel_rhs?

    nosensor_chan : channel_rhs

    stream_item : sensor_chan | nosensor_chan

    // A stream is an unordered sequence of fused channels, that can
    // optionally contain sensor specifications.

    stream : stream_item ("," stream_item)*

    %import common.DIGIT
    %import common.LETTER
    %import common.INT


You can think of a channel spec is that splitting the spec by "," gives groups
of channels that should be processed together and "late-fused".  Within each
group the "|" operator "early-fuses" the channels.

For instance, say we had a network and we wanted to process 3-channel rgb
images in one stream and 1-channel infrared images in a second stream and then
fuse them together. The channel specification for channels labled as 'red',
'green', 'blue', and 'infrared' would be:

.. code::

    infrared,red|green|blue


Sensors can be included with a colon prefix. Parenthesis can be used for
grouping.

.. code::


    S2:(infrared,red|green|blue)


.. |Pypi| image:: https://img.shields.io/pypi/v/delayed_image.svg
    :target: https://pypi.python.org/pypi/delayed_image

.. |PypiDownloads| image:: https://img.shields.io/pypi/dm/delayed_image.svg
    :target: https://pypistats.org/packages/delayed_image

.. |ReadTheDocs| image:: https://readthedocs.org/projects/delayed_image/badge/?version=latest
    :target: http://delayed_image.readthedocs.io/en/latest/

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/delayed_image/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/delayed_image/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/delayed_image/badges/main/coverage.svg
    :target: https://gitlab.kitware.com/computer-vision/delayed_image/commits/main

The delayed_image Module
========================

|Pypi| |PypiDownloads|


Ports the following from kwcoco:

* ChannelSpec
* SensorChannelSpec
* DelayedLoad & other delayed operations


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
        └─╼ Crop dsize=(130,131),space_slice=(slice(0,131,None),slice(1,131,None))
            └─╼ Warp dsize=(131,131),transform={scale=2.1000}
                └─╼ Warp dsize=(62,62),transform={scale=1.1000}
                    └─╼ Warp dsize=(56,56),transform={scale=1.1000}
                        └─╼ Warp dsize=(50,50),transform={scale=0.5000}
                            └─╼ Crop dsize=(99,100),space_slice=(slice(0,100,None),slice(1,100,None))
                                └─╼ Warp dsize=(100,100),transform={scale=0.5000}
                                    └─╼ Crop dsize=(199,200),space_slice=(slice(0,200,None),slice(1,200,None))
                                        └─╼ Warp dsize=(200,200),transform={scale=0.5000}
                                            └─╼ Crop dsize=(399,400),space_slice=(slice(0,400,None),slice(1,400,None))
                                                └─╼ Warp dsize=(621,621),transform={scale=1.1000}
                                                    └─╼ Warp dsize=(564,564),transform={scale=1.1000}
                                                        └─╼ Dequantize dsize=(512,512),quantization={quant_max=255,nodata=0}
                                                            └─╼ Load channels=r|g|b,dsize=(512,512),num_overviews=3,fname=astro_overviews=3.tif

    >>> # Optimize the chain
    >>> dopt = dimg.optimize()
    >>> dopt.write_network_text()
    ╙── Warp dsize=(128,130),transform={offset=(-0.6...,-1.0...),scale=1.5373}
        └─╼ Dequantize dsize=(80,83),quantization={quant_max=255,nodata=0}
            └─╼ Crop dsize=(80,83),space_slice=(slice(0,83,None),slice(3,83,None))
                └─╼ Overview dsize=(128,128),overview=2
                    └─╼ Load channels=r|g|b,dsize=(512,512),num_overviews=3,fname=astro_overviews=3.tif

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



SensorChanSpec
--------------


Includes the SensorChan spec, which makes handling channels from different
sensing sources easier.

It has a simple grammar:

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

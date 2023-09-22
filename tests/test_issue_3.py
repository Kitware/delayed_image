
def mwe_issue_3():
    """
    Ran into an issue in production where optimize was different between two
    very similar cases. The main difference was an extra take_channels
    operation, which seems to have caused optimize to fail.

    This appears to only be a problem when the underlying image data has
    overviews AND quantization is involved.

    The real world graphs pre and post optimize looked like:

    AC-SALIENT CROP RAW
    ╙── Warp dsize=(881,547),transform={offset=(81.0000,45.0000)},antialias=True,interpolation=linear,border_value=auto,noop_eps=0
        ╽
        Crop dsize=(800,502),space_slice=(slice(0,502,None),slice(0,800,None))
        ╽
        Warp dsize=(1140,554),transform={scale=(5.6701,5.7049)},antialias=True,interpolation=linear,border_value=auto,noop_eps=0
        ╽
        ChannelConcat axis=2
        ╽
        Crop channels=ac_salient,dsize=(201,97),space_slice=(slice(0,97,None),slice(0,201,None)),chan_idxs=[1]
        ╽
        Warp dsize=(201,97),transform={offset=(-0.0000,-0.0000),scale=(0.1764,0.1753)},antialias=True,interpolation=linear,border_value=auto,noop_eps=0
        ╽
        Dequantize dsize=(1140,554),quantization={orig_min=0,orig_max=1,quant_min=0,quant_max=32767,nodata=-9999}
        ╽
        Load channels=not_ac_salient|ac_salient,dsize=(1140,554),nodata_method=float,num_overviews=2,fname=crop_1459b622147d65ff.tif
    AC-CLASS CROP RAW
    ╙── Warp dsize=(881,547),transform={offset=(81.0000,45.0000)},antialias=True,interpolation=linear,border_value=auto,noop_eps=0
        ╽
        Crop dsize=(800,502),space_slice=(slice(0,502,None),slice(0,800,None))
        ╽
        Warp dsize=(1140,554),transform={scale=(5.6701,5.7049)},antialias=True,interpolation=linear,border_value=auto,noop_eps=0
        ╽
        ChannelConcat axis=2
        ╽
        Warp dsize=(201,97),transform={offset=(-0.0000,-0.0000),scale=(0.1764,0.1753)},antialias=True,interpolation=linear,border_value=auto,noop_eps=0
        ╽
        Dequantize dsize=(1140,554),quantization={orig_min=0,orig_max=1,quant_min=0,quant_max=32767,nodata=-9999}
        ╽
        Load channels=Site Preparation|Active Construction|Post Construction|No Activity,dsize=(1140,554),nodata_method=float,num_overviews=2,fname=crop_b54cf76afea0535b.tif
    AC-SALIENT CROP OPT
    ╙── ChannelConcat axis=2
        ╽
        Warp dsize=(881,547),transform={offset=(81.0000,45.0000)},antialias=True,interpolation=linear,border_value=auto,noop_eps=0
        ╽
        Dequantize dsize=(285,139),quantization={orig_min=0,orig_max=1,quant_min=0,quant_max=32767,nodata=-9999}
        ╽
        Crop channels=ac_salient,dsize=(285,139),space_slice=(slice(0,139,None),slice(0,285,None)),chan_idxs=[1]
        ╽
        Load channels=not_ac_salient|ac_salient,dsize=(1140,554),nodata_method=float,num_overviews=2,fname=crop_1459b622147d65ff.tif
    AC-CLASS CROP OPT
    ╙── ChannelConcat axis=2
        ╽
        Warp dsize=(881,547),transform={offset=(81.0000,45.0000)},antialias=True,interpolation=linear,border_value=auto,noop_eps=0
        ╽
        Dequantize dsize=(803,505),quantization={orig_min=0,orig_max=1,quant_min=0,quant_max=32767,nodata=-9999}
        ╽
        Crop dsize=(803,505),space_slice=(slice(0,505,None),slice(0,803,None))
        ╽
        Load channels=Site Preparation|Active Construction|Post Construction|No Activity,dsize=(1140,554),nodata_method=float,num_overviews=2,fname=crop__b54cf76afea0535b.tif
    """

    import delayed_image
    import kwimage
    from delayed_image import DelayedChannelConcat

    delayed_image.delayed_nodes.TRACE_OPTIMIZE = 0

    overviews = 3

    leaf = delayed_image.DelayedLoad.demo(channels='r|g|b',
                                          overviews=overviews,
                                          dsize=(1140, 554))
    leaf.prepare()

    sf1 = (0.1764, 0.1753)
    sf2 = (5.6701, 5.7049)

    sl = (slice(0, 502), slice(0, 800))

    quantization = {
        'orig_dtype': 'float32',
        'orig_min': 0,
        'orig_max': 1,
        'quant_min': 0,
        'quant_max': 255,
        'nodata': None,
    }

    def make_operation_tree(leaf, take_channels : bool):
        delayed = leaf
        delayed = delayed.scale(sf1)
        delayed = delayed.dequantize(quantization)
        if take_channels:
            delayed = delayed.take_channels('r')
        delayed = DelayedChannelConcat([delayed])
        delayed = delayed.scale(sf2)
        delayed = delayed.crop(sl, wrap=False, clip=False)
        return delayed

    delayed1 = make_operation_tree(leaf, take_channels=True)
    delayed2 = make_operation_tree(leaf, take_channels=False)

    opt1 = delayed1.optimize()
    if delayed_image.delayed_nodes.TRACE_OPTIMIZE:
        print('Opt Logs1')
        chain1 = [n for _, n in opt1._traverse()]
        for n in chain1:
            print(n._opt_logs)

    opt2 = delayed2.optimize()
    if delayed_image.delayed_nodes.TRACE_OPTIMIZE:
        print('Opt Logs2')
        chain2 = [n for _, n in opt2._traverse()]
        for n in chain2:
            print(n._opt_logs)

    print('')
    delayed1.print_graph()
    print('')
    delayed2.print_graph()

    print('')
    opt1.print_graph()
    print('')
    opt2.print_graph()

    # The optimized chains should have the same dsize AT EACH NODE!
    # The bug that this test was written for had the right dsize on the root of
    # the graph, but it was an intermediate node that caused the issue. So we
    # must enforce that the intermediate node (the quantize node in particular)
    # does not prematurely clip the data (and then get filled in with nans
    # based on the correct dsize later)
    chain1 = [n for _, n in opt1._traverse()]
    chain2 = [n for _, n in opt2._traverse()]

    # The class names will be the same (even though
    intermediate1 = [(n.__class__.__name__, n.dsize) for n in chain1]
    intermediate2 = [(n.__class__.__name__, n.dsize) for n in chain2]
    assert intermediate1 == intermediate2

    if 0:
        import kwplot
        kwplot.autoplt()
        im1 = delayed1.finalize()
        im2 = delayed2.finalize()
        im1 = kwimage.fill_nans_with_checkers(im1, on_value=0.3)
        im2 = kwimage.fill_nans_with_checkers(im2, on_value=0.3)
        kwplot.imshow(im1, fnum=1, pnum=(1, 2, 1))
        kwplot.imshow(im2, fnum=1, pnum=(1, 2, 2))

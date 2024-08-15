import kwarray
import kwimage
import numpy as np

# try:
#     from line_profiler import profile
# except ImportError:
#     from ubelt import identity as profile


# @profile
def test_shuffle_delayed_operations():
    """
    CommandLine:
        XDEV_PROFILE=1 xdoctest -m tests/test_delayed_ops.py test_shuffle_delayed_operations
    """
    # Try putting operations in differnet orders and ensure optimize always
    # fixes it.

    from delayed_image.delayed_leafs import DelayedLoad

    fpath = kwimage.grab_test_image_fpath()
    # overviews=3)
    base = DelayedLoad(fpath, channels='r|g|b')._load_metadata()
    quantization = {'quant_max': 255, 'nodata': 0}
    base.get_overview(1).dequantize(quantization).optimize()

    operations = [
        ('warp', {'scale': 1}),
        ('crop', (slice(None), slice(None))),
        ('get_overview', 1),
        ('dequantize', quantization),
    ]

    dequant_idx = [t[0] for t in operations].index('dequantize')

    # rng = kwarray.ensure_rng(None)
    rng = kwarray.ensure_rng(86159412070383637)

    # Repeat the test multiple times.
    num_times = 10
    for _ in range(num_times):
        num_ops = rng.randint(1, 30)
        op_idxs = rng.randint(0, len(operations), size=num_ops)

        # Don't allow dequantize more than once
        keep_flags = op_idxs != dequant_idx
        if not np.all(keep_flags):
            keeper = rng.choice(np.where(~keep_flags)[0])
            keep_flags[keeper] = True
        op_idxs = op_idxs[keep_flags]

        delayed = base
        for idx in op_idxs:
            name, args = operations[idx]
            func = getattr(delayed, name)
            delayed = func(args)

        # delayed.write_network_text(with_labels="name")
        opt = delayed.optimize()
        # opt.write_network_text(with_labels="name")

        # We always expect that we will get a sequence in the form
        expected_sequence = [
            'DelayedWarp', 'DelayedDequantize', 'DelayedCrop',
            'DelayedOverview', 'DelayedLoad'
        ]
        # But we are allowed to skip steps
        import networkx as nx
        graph = opt.as_graph()
        node_order = list(nx.topological_sort(graph))
        opname_order = [graph.nodes[n]['type'] for n in node_order]
        if opname_order[-1] != expected_sequence[-1]:
            raise AssertionError('Unexpected sequence')
        prev_idx = -1
        for opname in opname_order:
            this_idx = expected_sequence.index(opname)
            if this_idx <= prev_idx:
                raise AssertionError('Unexpected sequence')
            prev_idx = this_idx


def test_static_operation_optimize_single_chain():
    """
    There are 4 main operations:

        * warp - a general Affine transform (perhaps projective in the future)

        * crop - a slicing, translation, or band/channel sub-selection

        * get_overview - reads data from an "overview" which is effectively a
            precomputed downscale. It is efficient to replace downscales with
            overviews, but this does require careful modification of any other
            operation in the tree.

        * dequantize - converts integer quantized data back to its original
            floating point representation. It is important to do this operation
            before applying any sort of warping interpolation.
    """

    from delayed_image.delayed_leafs import DelayedLoad
    import kwimage

    try:
        import osgeo  # NOQA
    except ImportError:
        import pytest
        pytest.skip()

    # Grab a test image that contains 3 precomputed overviews
    fpath = kwimage.grab_test_image_fpath(overviews=3)

    # Start by pointing at an image on disk.
    base = DelayedLoad(fpath, channels='r|g|b')
    # Metadata about size / channels can be specified, but if it doesn't exist
    # prepare will read it from disk.
    base = base.prepare()

    # We can view the tree of operations at any time with print_graph
    base.print_graph(fields=True)
    """
    ╙── Load channels=r|g|b,dsize=(512,512),num_overviews=0,fname=astro.png
    """

    class mkslice:
        """ Helper to build slices """
        def __class_getitem__(self, index):
            return index
        def __getitem__(self, index):
            return index

    # A typical operation tree might be constructed like so
    delayed = base
    delayed = delayed.get_overview(1)
    delayed = delayed.scale(0.4)
    delayed = delayed.crop(mkslice()[0:1024, 0:1024], chan_idxs=[0, 2], clip=False, wrap=False)
    delayed = delayed.dequantize({
        'orig_min': 0, 'orig_max': 1,
        'quant_min': 0, 'quant_max': 255,
        'nodata': 0
    })
    delayed = delayed.warp(kwimage.Affine.random(rng=0))
    delayed = delayed.warp(kwimage.Affine.random(rng=1))
    delayed = delayed.warp(kwimage.Affine.random(rng=2))
    delayed = delayed.crop(mkslice()[0:32, 0:64], clip=False, wrap=False)

    # We can display the tree of operations as is like
    delayed.print_graph()
    """
    ╙── Crop dsize=(64,32),space_slice=(slice(0,32,None),slice(0,64,None))
        ╽
        Warp dsize=(4158,2765),transform={offset=(0.0851,-0.1109),scale=(1.3821,1.0222),shearx=-0.0000,theta=-0.0610}
        ╽
        Warp dsize=(2891,2710),transform={offset=(-0.9997,-0.3450),scale=(1.3647,1.6616),shearx=-0.0000,theta=-0.2739}
        ╽
        Warp dsize=(1621,1694),transform={offset=(0.1768,0.0769),scale=(1.4883,1.6561),shearx=0.0000,theta=-0.0585}
        ╽
        Dequantize dsize=(1024,1024),quantization={orig_min=0,orig_max=1,quant_min=0,quant_max=255,nodata=0}
        ╽
        Warp dsize=(1024,1024),transform={}
        ╽
        Crop dsize=(103,103),space_slice=(slice(0,103,None),slice(0,103,None))
        ╽
        Warp dsize=(103,103),transform={scale=0.4000}
        ╽
        Overview dsize=(256,256),overview=1
        ╽
        Load channels=r|g|b,dsize=(512,512),num_overviews=3,fname=astro_overviews=3.tif
    """

    # However, we have several places that could be replaced with more efficient operations.
    # * The linear warp operations can be fused together.
    # * The downscale operations can be transformed into overviews,
    # * And the crop operations can be moved as close to the data loading as
    # possible so the subsequent operations need to handle less data - as much of
    # the manipulated data will get cropped away.

    # The optimized tree looks like this
    optimized = delayed.optimize()
    optimized.print_graph()
    """
    ╙── Warp dsize=(64,32),transform={offset=(-0.6713,0.1755),scale=(0.5472,0.5773),shearx=0.1653,theta=-0.3208}
        ╽
        Dequantize dsize=(109,91),quantization={orig_min=0,orig_max=1,quant_min=0,quant_max=255,nodata=0}
        ╽
        Crop dsize=(109,91),space_slice=(slice(1,92,None),slice(0,109,None))
        ╽
        Load channels=r|g|b,dsize=(512,512),num_overviews=3,fname=astro_overviews=3.tif
    """

    # Notice that:
    # * All the overviews are moved into the load operation itself
    # * The crop is moved right after the load
    # * The dequantize happens before the warp
    # * There is only one final warp that happens at the very end.

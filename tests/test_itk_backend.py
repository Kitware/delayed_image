"""
SeeAlso:
    ~/code/delayed_image/examples/itk_backend.py
"""


def skip_if_missing_itk():
    try:
        import itk  # NOQA
    except ImportError:
        import pytest
        pytest.skip('requires itk to test')


def test_itk_warp():
    skip_if_missing_itk()

    from delayed_image import DelayedLoad
    self = DelayedLoad.demo().prepare()
    backend = 'itk'
    # backend = 'cv2'
    # dsize = (15, 15)
    dsize = 'auto'
    new = self.warp({'scale': 1 / 30}, backend=backend, dsize=dsize, antialias=0)
    new.print_graph(fields='all')
    opt = new.optimize()
    opt.print_graph(fields='all')
    result = opt.finalize()
    # import kwplot
    # kwplot.imshow(result)

    new2 = self.warp({'scale': 1 / 30}, backend='cv2', dsize=dsize, antialias=0)
    result2 = new2.finalize()
    # Results between backends should be very close
    assert (result2 - result).max() < 2


def fuzz_test_itk_warp():
    """
    Check that random transforms between different backends are very close to
    each other.
    """
    skip_if_missing_itk()

    import numpy as np
    from delayed_image import DelayedLoad
    import kwimage
    self = DelayedLoad.demo().prepare()

    import kwarray
    rng = kwarray.ensure_rng(414321)

    INTERACTIVE_DEBUG = 0
    if INTERACTIVE_DEBUG:
        import xdev
        num_trials = 1000
        trial_iter = list(range(num_trials))
        trial_iter = xdev.InteractiveIter(trial_iter)
    else:
        trial_iter = range(10)

    for _ in trial_iter:

        transform = kwimage.Affine.random(rng=rng)

        if rng.rand() > 0.9:
            transform = kwimage.Affine.translate(200) @ transform

        if rng.rand() > 0.9:
            transform = kwimage.Affine.translate(-300) @ transform

        if rng.rand() > 0.8:
            transform = kwimage.Affine.scale(0.01) @ transform

        if rng.rand() > 0.8:
            transform = kwimage.Affine.scale(0.1) @ transform

        dsize = 'auto'
        new1 = self.warp(transform, backend='itk', dsize=dsize, antialias=0)
        result1 = new1.finalize()

        new2 = self.warp(transform, backend='cv2', dsize=dsize, antialias=0)
        result2 = new2.finalize()

        # Results between backends should be very close
        delta = result2 - result1
        rmse = np.sqrt((delta ** 2).mean())
        assert rmse < 10, 'RMSE is above 10, this should be very unlikely to happen'
        print(f'rmse={rmse}')

        if INTERACTIVE_DEBUG:
            import kwplot
            kwplot.autompl()
            kwplot.imshow(result1, pnum=(1, 3, 1), fnum=1, doclf=1)
            kwplot.imshow(result2, pnum=(1, 3, 2), fnum=1)
            kwplot.imshow(np.abs(delta), pnum=(1, 3, 3), fnum=1)
            xdev.InteractiveIter.draw()

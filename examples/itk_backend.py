"""
SeeAlso:
    ~/code/delayed_image/tests/test_itk_backend.py
"""


def demo_itk_backend():
    """
    Warping has a cv2 and itk backend.
    """
    import numpy as np

    from delayed_image import DelayedLoad
    self = DelayedLoad.demo().prepare()

    param_grid = [
        {'transform': {'scale': 1 / 30}, 'antialias': 0},
        {'transform': {'scale': 1 / 30}, 'antialias': 1},
        {'transform': {'scale': 2.12}, 'antialias': 1},
        {'transform': {'scale': 2.12}, 'antialias': 0},
    ]

    results = []

    for warp_kwargs in param_grid:

        try:
            delayed_itk = self.warp(**warp_kwargs, backend='itk')
            result_itk = delayed_itk.finalize()
        except Exception:
            result_itk = None
            ...

        try:
            delayed_cv2 = self.warp(**warp_kwargs, backend='cv2')
            result_cv2 = delayed_cv2.finalize()
        except Exception:
            result_cv2 = None
            ...

        if result_itk is not None and result_cv2 is not None:
            # Results between backends should be very close
            delta = np.abs(result_itk.astype(float) - result_cv2.astype(float)).astype(np.uint8)
        else:
            delta = None

        row = {}
        row['warp_kwargs'] = warp_kwargs
        row['delta'] = delta
        row['result_cv2'] = result_cv2
        row['result_itk'] = result_itk
        results.append(row)

    import kwplot
    import kwimage
    kwplot.autompl()

    pnum_ = kwplot.PlotNums(nRows=len(results), nCols=1)
    fig = kwplot.figure(fnum=1, doclf=1, pnum_=pnum_[0])
    fig.clf()

    for row in results:
        result_itk = row['result_itk']
        result_cv2 = row['result_cv2']
        delta = row['delta']

        error_image = kwimage.draw_text_on_image({'width': result_cv2.shape[1], 'height': result_cv2.shape[0]}, 'X')

        if result_itk is None:
            result_itk = error_image
        if result_cv2 is None:
            result_cv2 = error_image
        if delta is None:
            delta = error_image
        result_itk = kwimage.draw_header_text(result_itk, 'itk')
        result_cv2 = kwimage.draw_header_text(result_cv2, 'cv2')
        delta = kwimage.draw_header_text(delta, 'delta')

        canvas = kwimage.stack_images([result_itk, result_cv2, delta], pad=3, axis=1)
        import ubelt as ub
        params = ub.urepr(row['warp_kwargs'], nl=0, compact=1, precision=3, nobr=1)
        kwplot.imshow(canvas, pnum=pnum_(), title=params)

    kwplot.show_if_requested()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/delayed_image/examples/itk_backend.py
    """
    demo_itk_backend()

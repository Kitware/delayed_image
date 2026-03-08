import numpy as np
import pytest


def test_lazy_gdal_multiband_overview_matches_reference(tmp_path):
    try:
        from osgeo import gdal  # NOQA
    except ImportError:
        pytest.skip('requires gdal')

    import kwimage
    from kwimage.im_io import _gdal_read
    from delayed_image.lazy_loaders import LazyGDalFrameFile

    rng = np.random.default_rng(0)
    data = rng.random((513, 517, 5), dtype=np.float32)
    fpath = tmp_path / 'odd_overview_multi.tif'
    kwimage.imwrite(fpath, data, backend='gdal', overviews=3)

    lazy = LazyGDalFrameFile(fpath)
    ds = lazy._ds

    cases = [
        (0, (slice(3, 55), slice(7, 61), slice(None))),
        (0, (slice(101, 170), slice(205, 260), [4, 0, 2])),
        (1, (slice(96, 211), slice(257, 259), slice(None))),
        (1, (slice(15, 88), slice(43, 127), [4, 0, 2])),
        (2, (slice(10, 56), slice(20, 89), [3])),
        (3, (slice(4, 50), slice(3, 55), [1, 4])),
    ]

    for overview, index in cases:
        reader = lazy.get_absolute_overview(overview)
        height, width, num_channels = reader.shape
        ypart, xpart, cpart = index
        y0 = 0 if ypart.start is None else ypart.start
        y1 = height if ypart.stop is None else ypart.stop
        x0 = 0 if xpart.start is None else xpart.start
        x1 = width if xpart.stop is None else xpart.stop
        if isinstance(cpart, list):
            band_indices = cpart
        else:
            band_indices = list(range(*cpart.indices(num_channels)))

        got = reader[index]
        gdalkw = {
            'xoff': x0,
            'yoff': y0,
            'win_xsize': x1 - x0,
            'win_ysize': y1 - y0,
        }
        want, _ = _gdal_read(
            gdal_dset=ds,
            overview=overview,
            nodata_method=None,
            nodata_value=None,
            ignore_color_table=True,
            band_indices=band_indices,
            gdalkw=gdalkw,
        )
        if want.ndim == 2:
            want = want[:, :, None]
        if got.ndim == 2:
            got = got[:, :, None]

        assert got.shape == want.shape
        assert np.allclose(got, want)

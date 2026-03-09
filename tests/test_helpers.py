import numpy as np


def test_dequantize_skips_unrepresentable_nodata_for_uint8_views():
    from delayed_image.helpers import dequantize

    band_major = np.arange(2 * 13 * 11, dtype=np.uint8).reshape(2, 13, 11)
    quant_data = band_major.transpose(1, 2, 0)
    assert not quant_data.flags['C_CONTIGUOUS']

    quantization = {
        'orig_dtype': 'float32',
        'orig_min': -94,
        'orig_max': 6040,
        'quant_min': 0,
        'quant_max': 32767,
        'nodata': -9999,
    }
    got = dequantize(quant_data, quantization)

    expected_quantization = dict(quantization)
    expected_quantization['nodata'] = None
    want = dequantize(quant_data, expected_quantization)

    assert np.isfinite(got).all()
    assert np.allclose(got, want)

import numpy as np
from unittest import mock


def test_warp_affine_matrix_mode_opencv_413_nearest_float64():
    from delayed_image import delayed_nodes

    with mock.patch.object(
        delayed_nodes, '_cv2_warp_affine_version', lambda: (4, 13, 0)
    ), mock.patch.object(delayed_nodes.kwimage, '__version__', '0.11.7'):
        delayed_nodes._WARP_AFFINE_KWIMAGE_WORKAROUND_WARNED = False

        assert delayed_nodes._warp_affine_matrix_mode(
            dtype=np.float64, backend='cv2', interpolation='nearest'
        ) == 'inverse'
        assert delayed_nodes._warp_affine_matrix_mode(
            dtype=np.float64, backend='auto', interpolation='nearest'
        ) == 'inverse'
        assert delayed_nodes._warp_affine_matrix_mode(
            dtype=np.float32, backend='cv2', interpolation='nearest'
        ) == 'forward'
        assert delayed_nodes._warp_affine_matrix_mode(
            dtype=np.float64, backend='cv2', interpolation='linear'
        ) == 'forward'
        assert delayed_nodes._warp_affine_matrix_mode(
            dtype=np.float64, backend='itk', interpolation='nearest'
        ) == 'forward'


def test_warp_affine_matrix_mode_kwimage_warning_once():
    from delayed_image import delayed_nodes

    import warnings

    with mock.patch.object(
        delayed_nodes, '_cv2_warp_affine_version', lambda: (4, 13, 0)
    ), mock.patch.object(delayed_nodes.kwimage, '__version__', '0.12.0'):
        delayed_nodes._WARP_AFFINE_KWIMAGE_WORKAROUND_WARNED = False

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float64, backend='cv2', interpolation='nearest'
            )
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float64, backend='cv2', interpolation='nearest'
            )

    assert len(records) == 1
    assert 'kwimage>=0.12.0' in str(records[0].message)

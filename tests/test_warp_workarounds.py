import numpy as np
from unittest import mock


def test_warp_affine_matrix_mode_opencv_413_nearest_float64():
    from delayed_image import delayed_nodes

    with mock.patch.object(
        delayed_nodes, '_cv2_warp_affine_version', lambda: (4, 13, 0)
    ), mock.patch.object(delayed_nodes.kwimage, '__version__', '0.11.7'):
        assert (
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float64, backend='cv2', interpolation='nearest'
            )
            == 'inverse'
        )
        assert (
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float64, backend='auto', interpolation='nearest'
            )
            == 'inverse'
        )
        assert (
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float32, backend='cv2', interpolation='nearest'
            )
            == 'forward'
        )
        assert (
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float64, backend='cv2', interpolation='linear'
            )
            == 'forward'
        )
        assert (
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float64, backend='itk', interpolation='nearest'
            )
            == 'forward'
        )


def test_warp_affine_matrix_mode_kwimage_012_disables_local_workaround():
    from delayed_image import delayed_nodes

    with mock.patch.object(
        delayed_nodes, '_cv2_warp_affine_version', lambda: (4, 13, 0)
    ), mock.patch.object(delayed_nodes.kwimage, '__version__', '0.12.0'):
        assert (
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float64, backend='cv2', interpolation='nearest'
            )
            == 'forward'
        )
        assert (
            delayed_nodes._warp_affine_matrix_mode(
                dtype=np.float64, backend='auto', interpolation='nearest'
            )
            == 'forward'
        )

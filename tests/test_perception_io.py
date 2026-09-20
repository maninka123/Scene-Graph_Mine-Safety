import numpy as np

from mine_safety.perception.io import voxelize


def test_voxel_features_follow_xyzrgb_contract():
    points = np.array([[1.0, 2.0, 3.0], [1.004, 2.004, 3.004]], dtype=np.float32)
    colours = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=np.float32)

    coordinates, features, inverse = voxelize(points, colours, voxel_size_m=0.01)

    assert coordinates.shape == (1, 3)
    np.testing.assert_allclose(features[0], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(inverse, [0, 0])

import numpy as np

from mine_safety.perception.anomaly import anomaly_nodes_from_probabilities, predictive_entropy


def test_predictive_entropy_orders_uncertainty():
    probabilities = np.array([[.99, .01], [.5, .5]])
    entropy = predictive_entropy(probabilities)
    assert entropy[1] > entropy[0]


def test_small_connected_component_is_removed_before_dbscan():
    large = np.array([(x, y, z) for x in range(3) for y in range(3) for z in range(3)])
    small = np.array([(x + 20, y, z) for x in range(2) for y in range(2) for z in range(2)])
    points = np.vstack([large, small]).astype(float) * 0.01
    probabilities = np.tile([0.5, 0.5], (len(points), 1))
    nodes = anomaly_nodes_from_probabilities(points, probabilities, epsilon_m=0.30)
    assert len(nodes) == 1
    assert nodes[0].voxel_count == len(large)

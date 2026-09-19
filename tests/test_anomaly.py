import numpy as np

from mine_safety.perception.anomaly import predictive_entropy


def test_predictive_entropy_orders_uncertainty():
    probabilities = np.array([[.99, .01], [.5, .5]])
    entropy = predictive_entropy(probabilities)
    assert entropy[1] > entropy[0]

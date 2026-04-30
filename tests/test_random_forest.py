import numpy as np
from mlpackage.supervised_learning import RandomForest

def test_random_forest_basic():
    X = np.array([[1, 2], [2, 3], [3, 3], [8, 8], [9, 9], [10, 10]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    rf = RandomForest(n_trees=5, max_depth=5)
    rf.fit(X, y)
    
    # Test predictions on seen data
    preds = rf.predict(X)
    assert np.array_equal(preds, y)
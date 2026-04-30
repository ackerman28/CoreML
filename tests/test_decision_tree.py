import numpy as np
from mlpackage.supervised_learning import DecisionTree

def test_decision_tree_simple():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 0, 1]) # Basically checking if it can split on feature index 0
    
    clf = DecisionTree(max_depth=10)
    clf.fit(X, y)
    predictions = clf.predict(X)
    
    assert np.array_equal(predictions, y)
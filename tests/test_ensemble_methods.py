import numpy as np
from mlpackage.supervised_learning import _BaseBagging, Perceptron

def test_base_bagging_with_perceptron():
    X = np.array([[1, 2], [2, 3], [8, 8], [9, 9]])
    y = np.array([0, 0, 1, 1])
    
    # Bagging a Perceptron
    ensemble = _BaseBagging(base_estimator=Perceptron(), n_estimators=5, random_state=42, mode='hard_vote')
    ensemble.fit(X, y)
    
    preds = ensemble.predict(X)
    assert len(preds) == 4
    assert np.array_equal(preds, y)
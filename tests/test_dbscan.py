import numpy as np
from mlpackage.unsupervised_learning import DBSCAN

def test_dbscan_moons():
    # Two simple clusters and one outlier
    X = np.array([[1, 1], [1.1, 1.1], [1.2, 1], 
                  [5, 5], [5.1, 5.1], [5.2, 5],
                  [10, 10]]) # Outlier
    
    db = DBSCAN(eps=1.0, min_samples=2)
    db.fit(X)
    labels = db.predict()
    
    # Check that we have two clusters and one noise point
    unique_labels = set(labels)
    assert 0 in unique_labels
    assert 1 in unique_labels
    assert -1 in unique_labels # Noise point (10, 10)
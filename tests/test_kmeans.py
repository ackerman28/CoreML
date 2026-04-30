import numpy as np
from mlpackage.unsupervised_learning import KMeans

def test_kmeans_clusters():
    # Create two very distinct blobs
    X = np.array([[1, 1], [1.1, 1.1], [10, 10], [10.1, 10.1]])
    
    km = KMeans(k=2, random_state=42) # Note: Adding random_state for stability
    km.fit(X)
    preds = km.predict(X)
    
    # Points 0,1 should have same label, 2,3 should have same label
    assert preds[0] == preds[1]
    assert preds[2] == preds[3]
    assert preds[0] != preds[2]
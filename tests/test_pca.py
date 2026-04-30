import numpy as np
from mlpackage.unsupervised_learning import PCA

def test_pca_dimensions():
    # Create a dummy dataset: 10 samples, 5 features
    X = np.random.rand(10, 5)
    n_comp = 2
    
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    X_transformed = pca.transform(X)
    
    # Check if the output shape is correct (10 samples, 2 features)
    assert X_transformed.shape == (10, 2)

def test_pca_reconstruction():
    # Data that is basically a line in 2D space
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    pca = PCA(n_components=1)
    pca.fit(X)
    X_transformed = pca.transform(X)
    
    # After reducing to 1D, the variance should be captured in one column
    assert X_transformed.shape == (4, 1)
    # The first and last points should be far apart in the new space
    assert np.abs(X_transformed[0] - X_transformed[-1]) > 1.0
import numpy as np

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray):
        # 1. Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Covariance matrix 
        # rowvar=False because columns are features
        cov = np.cov(X_centered, rowvar=False)

        # 3. Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # 4. Sort eigenvectors by eigenvalues descending
        # eigenvectors are returned as columns: eigenvectors[:, i]
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # 5. Store the top k components
        self.components = eigenvectors[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Project data onto the principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)
import numpy as np
from typing import Optional

class KMeans:
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4, random_state: Optional[int] = None):
        """
        K-Means Clustering implementation.
        
        Parameters:
        -----------
        k : int
            Number of clusters.
        max_iters : int
            Maximum number of iterations.
        tol : float
            Tolerance to declare convergence.
        random_state : int, optional
            Seed for reproducible centroid initialization.
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Calculates L2 distance between a point x1 and an array of points x2."""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def fit(self, X: np.ndarray):
        """Fits the model to the data using the EM-style algorithm."""
        n_samples, n_features = X.shape
        
        # 1. Initialize centroids randomly from existing data points
        rng = np.random.default_rng(self.random_state)
        random_indices = rng.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()

        for _ in range(self.max_iters):
            # 2. Assign Phase: Find nearest centroid for each point
            clusters = []
            for x in X:
                distances = self._euclidean_distance(x, self.centroids)
                clusters.append(np.argmin(distances))
            clusters = np.array(clusters)
            
            # 3. Update Phase: Calculate new mean of clusters
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                points_in_cluster = X[clusters == i]
                if len(points_in_cluster) > 0:
                    new_centroids[i] = np.mean(points_in_cluster, axis=0)
                else:
                    # If a cluster is empty, keep the previous centroid
                    new_centroids[i] = self.centroids[i]
            
            # 4. Convergence Check
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            
            self.centroids = new_centroids
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigns new data to the nearest existing cluster."""
        y_pred = []
        for x in X:
            distances = self._euclidean_distance(x, self.centroids)
            y_pred.append(np.argmin(distances))
        return np.array(y_pred)
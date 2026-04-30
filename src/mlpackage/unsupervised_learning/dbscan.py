import numpy as np

class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _get_neighbors(self, X, sample_idx):
        """Finds all points within distance eps of X[sample_idx]."""
        distances = np.linalg.norm(X - X[sample_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def fit(self, X: np.ndarray):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # Initialize all as noise (-1)
        cluster_id = 0
        
        visited = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i]:
                continue
            
            visited[i] = True
            neighbors = self._get_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                # Label as noise for now
                self.labels_[i] = -1
            else:
                # Start a new cluster
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1
        return self

    def _expand_cluster(self, X, sample_idx, neighbors, cluster_id, visited):
        """Recursively expands the cluster using a Breadth-First approach."""
        self.labels_[sample_idx] = cluster_id
        
        # We use a list as a queue to explore neighbors
        queue = list(neighbors)
        
        while queue:
            neighbor_idx = queue.pop(0)
            
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._get_neighbors(X, neighbor_idx)
                
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)
            
            # If point isn't assigned to a cluster yet, assign it
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id

    def predict(self, X=None):
        """Returns the labels generated during fit."""
        return self.labels_
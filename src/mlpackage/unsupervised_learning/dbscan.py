import numpy as np
from typing import Optional


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    DBSCAN groups together points that are closely packed (high density)
    and marks points in low-density regions as outliers. Unlike K-Means,
    it does not require specifying the number of clusters in advance and
    can discover clusters of arbitrary shape.

    The algorithm classifies each point as one of three types:

    - **Core point** — has at least ``min_samples`` points within ``eps``.
    - **Border point** — within ``eps`` of a core point but not itself a core point.
    - **Noise point** — not within ``eps`` of any core point. Labeled ``-1``.

    Parameters
    ----------
    eps : float, optional (default=0.5)
        Maximum distance between two samples to be considered neighbors.
        Controls the size of each point's neighborhood.
    min_samples : int, optional (default=5)
        Minimum number of points in a neighborhood for a point to be
        considered a core point (including itself).
    metric : str, optional (default='euclidean')
        Distance metric. One of ``'euclidean'`` (L2) or ``'manhattan'`` (L1).

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster label for each point. Noise points are labeled ``-1``.
        Cluster labels start from ``0``.
    core_sample_indices_ : np.ndarray
        Indices of core points identified during ``fit()``.
    n_clusters_ : int
        Number of clusters found (excluding noise).
    n_noise_ : int
        Number of points labeled as noise.

    Examples
    --------
    >>> from mlpackage.unsupervised_learning.dbscan import DBSCAN
    >>> import numpy as np
    >>> X = np.array([[1,1],[1.1,1],[5,5],[5.1,5],[10,10]], dtype=float)
    >>> db = DBSCAN(eps=0.5, min_samples=2)
    >>> db.fit(X)
    >>> db.labels_
    array([ 0,  0,  1,  1, -1])
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = "euclidean"):
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(
                f"metric must be 'euclidean' or 'manhattan', got '{metric}'"
            )

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_: np.ndarray = None
        self.core_sample_indices_: np.ndarray = None
        self.n_clusters_: int = None
        self.n_noise_: int = None

    def _compute_distances(self, X: np.ndarray,
                            idx: int) -> np.ndarray:
        """
        Compute distances from X[idx] to all other points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        idx : int
            Index of the query point.

        Returns
        -------
        distances : np.ndarray of shape (n_samples,)
        """
        if self.metric == "euclidean":
            return np.linalg.norm(X - X[idx], axis=1)
        else:
            return np.sum(np.abs(X - X[idx]), axis=1)

    def _get_neighbors(self, X: np.ndarray, idx: int) -> np.ndarray:
        """
        Find all points within eps of X[idx].

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        idx : int
            Index of the query point.

        Returns
        -------
        neighbor_indices : np.ndarray
            Indices of points within eps (including idx itself).
        """
        distances = self._compute_distances(X, idx)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X: np.ndarray, idx: int,
                         neighbors: np.ndarray, cluster_id: int,
                         visited: np.ndarray) -> None:
        """
        Grow a cluster from a core point using BFS.

        Iteratively adds reachable points to the cluster. A point is
        density-reachable if it lies within eps of a core point already
        in the cluster.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        idx : int
            Index of the seed core point.
        neighbors : np.ndarray
            Initial neighborhood of idx.
        cluster_id : int
            Integer label for the current cluster.
        visited : np.ndarray of bool
            Tracks which points have been processed.
        """
        self.labels_[idx] = cluster_id
        queue = list(neighbors)

        while queue:
            current = queue.pop(0)

            if not visited[current]:
                visited[current] = True
                new_neighbors = self._get_neighbors(X, current)
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors.tolist())

            if self.labels_[current] == -1:
                self.labels_[current] = cluster_id

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Run DBSCAN on the input data.

        Iterates over all points. For each unvisited point, retrieves its
        neighborhood. Core points seed a new cluster which is grown via BFS.
        Non-core points are initially marked as noise (``-1``) and may later
        be absorbed as border points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : DBSCAN
            Fitted estimator.

        Raises
        ------
        ValueError
            If X is not a 2D array.
        """
        X = np.array(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")

        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

        self.core_sample_indices_ = np.array([
            i for i in range(n_samples)
            if len(self._get_neighbors(X, i)) >= self.min_samples
        ])
        self.n_clusters_ = cluster_id
        self.n_noise_    = int(np.sum(self.labels_ == -1))

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and return cluster labels in one step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels. Noise points are labeled ``-1``.
        """
        return self.fit(X).labels_
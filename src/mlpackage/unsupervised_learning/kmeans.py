import numpy as np
from typing import Optional


class KMeans:
    """
    K-Means clustering via the Lloyd's algorithm (EM-style).

    Partitions $n$ samples into $k$ clusters by alternating between:

    1. **Assignment step** — assign each sample to the nearest centroid.
    2. **Update step** — recompute each centroid as the mean of its assigned samples.

    Supports two initialization strategies: random sampling from the data
    (``'random'``) and the smarter **K-Means++** seeding (``'kmeans++'``),
    which spreads initial centroids apart to improve convergence and reduce
    sensitivity to initialization.

    Parameters
    ----------
    k : int, optional (default=3)
        Number of clusters.
    max_iters : int, optional (default=300)
        Maximum number of EM iterations.
    tol : float, optional (default=1e-4)
        Convergence tolerance. Stops when centroid shift < tol.
    init : str, optional (default='kmeans++')
        Centroid initialization strategy. One of ``'random'`` or ``'kmeans++'``.
    n_init : int, optional (default=10)
        Number of independent runs. The best result (lowest inertia) is kept.
    random_state : int or None, optional (default=None)
        Seed for reproducibility.

    Attributes
    ----------
    centroids_ : np.ndarray of shape (k, n_features)
        Final centroid positions after fitting.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster assignment for each training sample.
    inertia_ : float
        Sum of squared distances of samples to their nearest centroid (WCSS).
    n_iter_ : int
        Number of iterations run in the best trial.

    Examples
    --------
    >>> from mlpackage.unsupervised_learning.kmeans import KMeans
    >>> import numpy as np
    >>> X = np.array([[1,1],[1.1,1.1],[10,10],[10.1,10.1]], dtype=float)
    >>> km = KMeans(k=2, random_state=42)
    >>> km.fit(X)
    >>> km.labels_
    array([0, 0, 1, 1])
    """

    def __init__(self, k: int = 3, max_iters: int = 300, tol: float = 1e-4,
                 init: str = "kmeans++", n_init: int = 10,
                 random_state: Optional[int] = None):
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}")
        if init not in ("random", "kmeans++"):
            raise ValueError(f"init must be 'random' or 'kmeans++', got '{init}'")
        if n_init <= 0:
            raise ValueError(f"n_init must be positive, got {n_init}")

        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.centroids_: np.ndarray = None
        self.labels_: np.ndarray = None
        self.inertia_: float = None
        self.n_iter_: int = None

    def _init_centroids_random(self, X: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
        """
        Initialize centroids by sampling k distinct data points uniformly.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        rng : np.random.Generator

        Returns
        -------
        np.ndarray of shape (k, n_features)
        """
        idxs = rng.choice(X.shape[0], size=self.k, replace=False)
        return X[idxs].copy()

    def _init_centroids_kmeans_plus_plus(self, X: np.ndarray,
                                          rng: np.random.Generator) -> np.ndarray:
        """
        K-Means++ centroid initialization.

        Selects centroids sequentially with probability proportional to the
        squared distance from the nearest already-chosen centroid. This
        spreads initial centroids and reduces the chance of poor convergence.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        rng : np.random.Generator

        Returns
        -------
        np.ndarray of shape (k, n_features)
        """
        first_idx = rng.integers(0, X.shape[0])
        centroids = [X[first_idx].copy()]

        for _ in range(1, self.k):
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs = dists / dists.sum()
            next_idx = rng.choice(X.shape[0], p=probs)
            centroids.append(X[next_idx].copy())

        return np.array(centroids)

    def _assign_clusters(self, X: np.ndarray,
                          centroids: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid (Euclidean distance).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        centroids : np.ndarray of shape (k, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray,
                           old_centroids: np.ndarray) -> np.ndarray:
        """
        Recompute centroids as the mean of assigned samples.

        Empty clusters retain their previous centroid position.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        labels : np.ndarray of shape (n_samples,)
        old_centroids : np.ndarray of shape (k, n_features)

        Returns
        -------
        new_centroids : np.ndarray of shape (k, n_features)
        """
        new_centroids = np.zeros_like(old_centroids)
        for i in range(self.k):
            members = X[labels == i]
            new_centroids[i] = np.mean(members, axis=0) if len(members) > 0 else old_centroids[i]
        return new_centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                          centroids: np.ndarray) -> float:
        """
        Compute within-cluster sum of squares (WCSS / inertia).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        labels : np.ndarray of shape (n_samples,)
        centroids : np.ndarray of shape (k, n_features)

        Returns
        -------
        float
            Total inertia.
        """
        return float(np.sum([
            np.sum((X[labels == i] - centroids[i]) ** 2)
            for i in range(self.k)
        ]))

    def _single_run(self, X: np.ndarray,
                     rng: np.random.Generator) -> tuple:
        """
        Run one complete K-Means trial from initialization to convergence.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        rng : np.random.Generator

        Returns
        -------
        centroids : np.ndarray of shape (k, n_features)
        labels : np.ndarray of shape (n_samples,)
        inertia : float
        n_iter : int
        """
        if self.init == "kmeans++":
            centroids = self._init_centroids_kmeans_plus_plus(X, rng)
        else:
            centroids = self._init_centroids_random(X, rng)

        for iteration in range(1, self.max_iters + 1):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels, centroids)
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit K-Means by running n_init independent trials and keeping the best.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : KMeans
            Fitted estimator.

        Raises
        ------
        ValueError
            If k exceeds the number of samples.
        """
        X = np.array(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if self.k > X.shape[0]:
            raise ValueError(
                f"k ({self.k}) cannot exceed number of samples ({X.shape[0]})"
            )

        best_inertia = np.inf
        best_centroids, best_labels, best_n_iter = None, None, None
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._single_run(X, rng)
            if inertia < best_inertia:
                best_inertia   = inertia
                best_centroids = centroids
                best_labels    = labels
                best_n_iter    = n_iter

        self.centroids_ = best_centroids
        self.labels_    = best_labels
        self.inertia_   = best_inertia
        self.n_iter_    = best_n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new samples to the nearest fitted centroid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)

        Raises
        ------
        RuntimeError
            If called before fitting.
        """
        if self.centroids_ is None:
            raise RuntimeError("KMeans is not fitted. Call fit() before predict().")
        X = np.array(X, dtype=float)
        return self._assign_clusters(X, self.centroids_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels in one step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        return self.fit(X).labels_
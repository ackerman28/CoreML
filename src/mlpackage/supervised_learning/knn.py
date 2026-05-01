import numpy as np
from collections import Counter


class KNN:
    """
    K-Nearest Neighbors classifier and regressor.

    A non-parametric, instance-based learning algorithm that makes predictions
    by finding the K training samples closest to a query point and aggregating
    their labels (classification) or values (regression).

    KNN performs no explicit training — it memorizes the training set and
    defers all computation to prediction time (lazy learning).

    Parameters
    ----------
    k : int, optional (default=3)
        Number of nearest neighbors to consider. Larger K produces smoother
        decision boundaries but may underfit; smaller K is more flexible but
        sensitive to noise.
    metric : str, optional (default='euclidean')
        Distance metric to use. One of:
        - ``'euclidean'`` : L2 norm — standard straight-line distance.
        - ``'manhattan'`` : L1 norm — sum of absolute differences.
        - ``'minkowski'`` : generalized Lp norm (requires setting ``p``).
    p : int, optional (default=2)
        Power parameter for the Minkowski metric.
        p=1 is equivalent to Manhattan; p=2 is equivalent to Euclidean.
    task : str, optional (default='classification')
        Whether to perform ``'classification'`` (majority vote) or
        ``'regression'`` (mean of neighbor values).
    weights : str, optional (default='uniform')
        How to weight neighbor contributions. One of:
        - ``'uniform'`` : all neighbors contribute equally.
        - ``'distance'`` : closer neighbors are weighted more heavily
          (weight = 1 / distance).

    Attributes
    ----------
    X_train : np.ndarray of shape (n_samples, n_features)
        Stored training feature matrix.
    y_train : np.ndarray of shape (n_samples,)
        Stored training labels or target values.

    Examples
    --------
    >>> from mlpackage.supervised_learning.knn import KNN
    >>> import numpy as np
    >>> X_train = np.array([[0,0],[1,1],[5,5],[6,6]], dtype=float)
    >>> y_train = np.array([0, 0, 1, 1])
    >>> clf = KNN(k=3)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(np.array([[0.5, 0.5], [5.5, 5.5]]))
    array([0, 1])
    """

    def __init__(self, k: int = 3, metric: str = "euclidean",
                 p: int = 2, task: str = "classification",
                 weights: str = "uniform"):
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError(
                f"metric must be one of 'euclidean', 'manhattan', 'minkowski', got '{metric}'"
            )
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got '{task}'"
            )
        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got '{weights}'"
            )

        self.k = k
        self.metric = metric
        self.p = p
        self.task = task
        self.weights = weights
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """
        Store the training data (lazy learning — no computation performed).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Training labels (classification) or target values (regression).

        Returns
        -------
        self : KNN
            Fitted estimator (enables method chaining).

        Raises
        ------
        ValueError
            If X and y have inconsistent numbers of samples, or if k exceeds
            the number of training samples.
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        if self.k > X.shape[0]:
            raise ValueError(
                f"k ({self.k}) cannot exceed the number of training samples ({X.shape[0]})"
            )

        self.X_train = X
        self.y_train = y
        return self

    def _compute_distance(self, x: np.ndarray) -> np.ndarray:
        """
        Compute distances from a query point to all training samples.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Single query point.

        Returns
        -------
        distances : np.ndarray of shape (n_train_samples,)
            Distance from x to each training sample.
        """
        if self.metric == "euclidean":
            return np.linalg.norm(self.X_train - x, axis=1)
        elif self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:  # minkowski
            return np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1 / self.p)

    def _predict_single(self, x: np.ndarray):
        """
        Predict the label or value for a single query point.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Single query point.

        Returns
        -------
        int or float
            Predicted class label (classification) or mean value (regression).
        """
        distances = self._compute_distance(x)
        k_indices = np.argsort(distances)[:self.k]
        k_distances = distances[k_indices]
        k_labels = self.y_train[k_indices]

        if self.weights == "uniform":
            if self.task == "classification":
                return Counter(k_labels).most_common(1)[0][0]
            else:
                return float(np.mean(k_labels))
        else:
            # Distance weighting — handle exact matches (distance = 0)
            if np.any(k_distances == 0):
                zero_mask = k_distances == 0
                if self.task == "classification":
                    return Counter(k_labels[zero_mask]).most_common(1)[0][0]
                else:
                    return float(np.mean(k_labels[zero_mask]))
            w = 1 / k_distances
            w /= w.sum()
            if self.task == "classification":
                weighted_votes = {}
                for label, weight in zip(k_labels, w):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                return max(weighted_votes, key=weighted_votes.get)
            else:
                return float(np.dot(w, k_labels))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels or values for a batch of samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels or target values.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted.
        """
        if self.X_train is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy (classification) or R² (regression) on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            True labels or target values.

        Returns
        -------
        float
            Accuracy for classification; R² for regression.
        """
        y = np.array(y)
        y_pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(y_pred == y))
        else:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
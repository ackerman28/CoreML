import numpy as np
import copy
from collections import Counter
from typing import Any, Optional


def _aggregate_predictions(y_preds: np.ndarray, mode: str) -> np.ndarray:
    """
    Aggregate predictions from multiple estimators.

    Parameters
    ----------
    y_preds : np.ndarray of shape (n_samples, n_estimators)
        Predictions from each estimator for each sample.
    mode : str
        Aggregation strategy. One of:
        - ``'hard_vote'`` : majority class label (classification).
        - ``'average'``   : mean prediction (regression).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Aggregated predictions.

    Raises
    ------
    ValueError
        If mode is not recognized.
    """
    if mode == "hard_vote":
        def majority(row):
            counts = Counter(row)
            max_count = max(counts.values())
            return min(k for k, v in counts.items() if v == max_count)
        return np.array([majority(y_preds[i]) for i in range(y_preds.shape[0])])
    elif mode == "average":
        return np.mean(y_preds, axis=1)
    raise ValueError(f"Unknown aggregation mode: '{mode}'. Use 'hard_vote' or 'average'.")


class _BaseBagging:
    """
    Bootstrap Aggregating (Bagging) base class.

    Each estimator is trained on an independent bootstrap sample (sampling
    with replacement). Predictions are aggregated via majority vote
    (classification) or averaging (regression).

    Bagging reduces variance without significantly increasing bias, making
    it especially effective for high-variance base learners like decision trees.

    Parameters
    ----------
    base_estimator : object
        Unfitted estimator with a ``fit(X, y)`` and ``predict(X)`` interface.
        A deep copy is made for each bootstrap round.
    n_estimators : int, optional (default=10)
        Number of estimators to train.
    random_state : int or None, optional (default=None)
        Seed for reproducible bootstrap sampling.
    mode : str, optional (default='hard_vote')
        Aggregation mode. ``'hard_vote'`` for classification,
        ``'average'`` for regression.

    Attributes
    ----------
    estimators_ : list
        Fitted estimator instances after calling ``fit()``.
    """

    def __init__(self, base_estimator: Any, n_estimators: int = 10,
                 random_state: Optional[int] = None, mode: str = "hard_vote"):
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}")
        if mode not in ("hard_vote", "average"):
            raise ValueError(f"mode must be 'hard_vote' or 'average', got '{mode}'")

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.mode = mode
        self.estimators_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseBagging":
        """
        Train n_estimators on independent bootstrap samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target labels or values.

        Returns
        -------
        self : _BaseBagging
            Fitted ensemble.
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        n_samples = X.shape[0]
        self.estimators_ = []
        
        # Use RandomState for better reproducibility with integer seeds
        rng = np.random.RandomState(self.random_state)
        
        for _ in range(self.n_estimators):
            estimator = copy.deepcopy(self.base_estimator)
            
            # If base estimator has random_state, set it for reproducibility
            if self.random_state is not None and hasattr(estimator, 'random_state'):
                estimator.random_state = rng.randint(0, 2**31)
            
            # Generate bootstrap indices
            idxs = rng.choice(n_samples, size=n_samples, replace=True)
            estimator.fit(X[idxs], y[idxs])
            self.estimators_.append(estimator)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate predictions from all estimators.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Aggregated predictions.

        Raises
        ------
        RuntimeError
            If called before fitting.
        """
        if not self.estimators_:
            raise RuntimeError("Ensemble is not fitted. Call fit() before predict().")
        X = np.array(X, dtype=float)
        all_preds = np.array([est.predict(X) for est in self.estimators_]).T
        return _aggregate_predictions(all_preds, self.mode)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy (classification) or R² (regression).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        float
        """
        y = np.array(y)
        y_pred = self.predict(X)
        if self.mode == "average":
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        return float(np.mean(y_pred == y))


class BaggingClassifier(_BaseBagging):
    """
    Bagging ensemble for classification tasks.

    Wraps ``_BaseBagging`` with ``mode='hard_vote'`` and provides a
    ``predict_proba()`` method returning class vote fractions.

    Parameters
    ----------
    base_estimator : object
        Unfitted classifier with ``fit`` / ``predict`` interface.
    n_estimators : int, optional (default=10)
        Number of bootstrap estimators.
    random_state : int or None, optional (default=None)
        Seed for reproducibility.

    Examples
    --------
    >>> from mlpackage.supervised_learning.ensemble import BaggingClassifier
    >>> from mlpackage.supervised_learning.decision_tree import DecisionTree
    >>> import numpy as np
    >>> X = np.array([[1,2],[2,3],[8,8],[9,9]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = BaggingClassifier(DecisionTree(max_depth=3), n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    >>> clf.predict(X)
    array([0, 0, 1, 1])
    """

    def __init__(self, base_estimator: Any, n_estimators: int = 10,
                 random_state: Optional[int] = None):
        super().__init__(base_estimator, n_estimators, random_state, mode="hard_vote")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate class probabilities as the fraction of estimators voting for each class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class vote fractions for each sample.
        """
        if not self.estimators_:
            raise RuntimeError("Ensemble is not fitted. Call fit() before predict_proba().")
        X = np.array(X, dtype=float)
        all_preds = np.array([est.predict(X) for est in self.estimators_]).T
        classes = np.unique(all_preds)
        proba = np.zeros((X.shape[0], len(classes)))
        for i, cls in enumerate(classes):
            proba[:, i] = np.mean(all_preds == cls, axis=1)
        return proba


class BaggingRegressor(_BaseBagging):
    """
    Bagging ensemble for regression tasks.

    Wraps ``_BaseBagging`` with ``mode='average'``.

    Parameters
    ----------
    base_estimator : object
        Unfitted regressor with ``fit`` / ``predict`` interface.
    n_estimators : int, optional (default=10)
        Number of bootstrap estimators.
    random_state : int or None, optional (default=None)
        Seed for reproducibility.
    """

    def __init__(self, base_estimator: Any, n_estimators: int = 10,
                 random_state: Optional[int] = None):
        super().__init__(base_estimator, n_estimators, random_state, mode="average")


class VotingClassifier:
    """
    Hard or soft voting ensemble over a fixed set of diverse classifiers.

    Unlike Bagging — which uses many copies of one base estimator — the
    Voting Classifier combines predictions from several *different* models.
    This leverages the complementary strengths of each model.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Named estimators to include in the ensemble.
    voting : str, optional (default='hard')
        ``'hard'`` uses majority class label; ``'soft'`` averages
        ``predict_proba()`` outputs (each estimator must support it).

    Attributes
    ----------
    estimators_ : list
        Fitted estimator instances.

    Examples
    --------
    >>> from mlpackage.supervised_learning.ensemble import VotingClassifier
    >>> from mlpackage.supervised_learning.decision_tree import DecisionTree
    >>> from mlpackage.supervised_learning.knn import KNN
    >>> clf = VotingClassifier([('dt', DecisionTree()), ('knn', KNN())])
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    """

    def __init__(self, estimators: list, voting: str = "hard"):
        if voting not in ("hard", "soft"):
            raise ValueError(f"voting must be 'hard' or 'soft', got '{voting}'")
        if len(estimators) == 0:
            raise ValueError("estimators list cannot be empty")

        self.estimators = estimators
        self.voting = voting
        self.estimators_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingClassifier":
        """
        Fit all estimators on the full training set.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        self : VotingClassifier
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.estimators_ = []
        for _, estimator in self.estimators:
            clf = copy.deepcopy(estimator)
            clf.fit(X, y)
            self.estimators_.append(clf)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels via hard or soft voting.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        if not self.estimators_:
            raise RuntimeError("VotingClassifier is not fitted. Call fit() first.")
        X = np.array(X, dtype=float)

        if self.voting == "hard":
            all_preds = np.array([est.predict(X) for est in self.estimators_]).T
            return _aggregate_predictions(all_preds, "hard_vote")
        else:
            probas = np.array([est.predict_proba(X) for est in self.estimators_])
            avg_proba = np.mean(probas, axis=0)
            return np.argmax(avg_proba, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(X) == np.array(y)))
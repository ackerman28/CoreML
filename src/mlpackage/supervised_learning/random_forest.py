import numpy as np
from collections import Counter
from .decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest ensemble classifier and regressor.

    Builds a collection of Decision Trees, each trained on a bootstrap sample
    of the data with a random subset of features considered at each split.
    Predictions are aggregated via majority vote (classification) or mean
    (regression).

    Two sources of randomness reduce variance and decorrelate the trees:

    1. **Bootstrap sampling** — each tree sees a random sample with replacement.
    2. **Feature subsampling** — each split considers only a random subset of
       features (default: ``sqrt(n_features)`` for classification).

    Parameters
    ----------
    n_trees : int, optional (default=100)
        Number of trees in the forest. More trees reduce variance but
        increase training time.
    max_depth : int, optional (default=10)
        Maximum depth of each individual tree.
    min_samples_split : int, optional (default=2)
        Minimum samples required to split a node in each tree.
    n_features : int or None, optional (default=None)
        Number of features to consider at each split. If None, defaults to
        ``int(sqrt(n_features))``.
    criterion : str, optional (default='entropy')
        Impurity measure for splits. One of ``'entropy'`` or ``'gini'``.
    task : str, optional (default='classification')
        One of ``'classification'`` or ``'regression'``.
    random_state : int or None, optional (default=None)
        Seed for reproducibility of bootstrap sampling and tree construction.

    Attributes
    ----------
    trees : list of DecisionTree
        The fitted individual trees.
    feature_importances_ : np.ndarray of shape (n_features,)
        Mean feature importances averaged across all trees.
    oob_score_ : float or None
        Out-of-bag accuracy (classification) or R² (regression), computed
        if ``oob_score=True`` during fit.

    Examples
    --------
    >>> from mlpackage.supervised_learning.random_forest import RandomForest
    >>> import numpy as np
    >>> X = np.array([[1,2],[2,3],[8,8],[9,9]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> rf = RandomForest(n_trees=10, random_state=42)
    >>> rf.fit(X, y)
    >>> rf.predict(X)
    array([0, 0, 1, 1])
    """

    def __init__(self, n_trees: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, n_features: int = None,
                 criterion: str = "entropy", task: str = "classification",
                 random_state: int = None):
        if n_trees <= 0:
            raise ValueError(f"n_trees must be a positive integer, got {n_trees}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if criterion not in ("entropy", "gini"):
            raise ValueError(f"criterion must be 'entropy' or 'gini', got '{criterion}'")
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.criterion = criterion
        self.task = task
        self.random_state = random_state
        self.trees: list = []
        self.feature_importances_: np.ndarray = None
        self.oob_score_: float = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray,
            oob_score: bool = False) -> "RandomForest":
        """
        Build the forest by training trees on bootstrap samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target labels or values.
        oob_score : bool, optional (default=False)
            Whether to compute the out-of-bag score after fitting.

        Returns
        -------
        self : RandomForest
            Fitted estimator.

        Raises
        ------
        ValueError
            If X and y have inconsistent numbers of samples.
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

        n_samples, n_feats = X.shape
        n_features_split = self.n_features or max(1, int(np.sqrt(n_feats)))

        self.trees = []
        oob_predictions = [[] for _ in range(n_samples)]

        for i in range(self.n_trees):
            seed = None if self.random_state is None else self.random_state + i
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_features_split,
                criterion=self.criterion,
                task=self.task,
                random_state=seed
            )
            boot_idxs = self._bootstrap_indices(n_samples)
            tree.fit(X[boot_idxs], y[boot_idxs])
            self.trees.append(tree)

            if oob_score:
                oob_idxs = np.setdiff1d(np.arange(n_samples), boot_idxs)
                preds = tree.predict(X[oob_idxs])
                for idx, pred in zip(oob_idxs, preds):
                    oob_predictions[idx].append(pred)

        # Average feature importances across all trees
        importances = np.zeros(n_feats)
        for tree in self.trees:
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
        self.feature_importances_ = importances / self.n_trees

        if oob_score:
            self.oob_score_ = self._compute_oob_score(y, oob_predictions)

        return self

    def _bootstrap_indices(self, n_samples: int) -> np.ndarray:
        """
        Draw bootstrap sample indices (with replacement).

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Bootstrap sample indices.
        """
        return self._rng.choice(n_samples, size=n_samples, replace=True)

    def _compute_oob_score(self, y: np.ndarray,
                           oob_predictions: list) -> float:
        """
        Compute out-of-bag score from accumulated OOB predictions.

        Parameters
        ----------
        y : np.ndarray
            True target values.
        oob_predictions : list of lists
            Accumulated predictions for each sample from trees that
            did not include it in their bootstrap sample.

        Returns
        -------
        float
            OOB accuracy (classification) or R² (regression).
        """
        oob_preds = []
        valid_idxs = []
        for i, preds in enumerate(oob_predictions):
            if len(preds) > 0:
                if self.task == "classification":
                    oob_preds.append(Counter(preds).most_common(1)[0][0])
                else:
                    oob_preds.append(np.mean(preds))
                valid_idxs.append(i)

        if not valid_idxs:
            return None

        y_true = y[valid_idxs]
        y_oob  = np.array(oob_preds)

        if self.task == "classification":
            return float(np.mean(y_oob == y_true))
        else:
            ss_res = np.sum((y_true - y_oob) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate predictions from all trees.

        Classification uses majority vote; regression uses the mean.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels or values.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted.
        """
        if not self.trees:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        X = np.array(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        if self.task == "regression":
            return np.mean(tree_preds, axis=0)

        tree_preds = tree_preds.T
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate class probabilities as the fraction of trees voting for each class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probability estimates.

        Raises
        ------
        RuntimeError
            If called on a regression forest or before fitting.
        """
        if not self.trees:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict_proba().")
        if self.task == "regression":
            raise RuntimeError("predict_proba() is not available for regression forests.")
        X = np.array(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        classes = np.unique(tree_preds)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(classes)))
        for i, cls in enumerate(classes):
            proba[:, i] = np.mean(tree_preds == cls, axis=0)
        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy (classification) or R² (regression).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            True labels or values.

        Returns
        -------
        float
            Accuracy for classification; R² for regression.
        """
        y = np.array(y)
        y_pred = self.predict(X)
        if self.task == "regression":
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        return float(np.mean(y_pred == y))
import numpy as np
from collections import Counter


class Node:
    """
    A single node in a Decision Tree.

    Parameters
    ----------
    feature : int or None
        Index of the feature used for splitting. None for leaf nodes.
    threshold : float or None
        Threshold value for the split. None for leaf nodes.
    left : Node or None
        Left child node (samples where feature <= threshold).
    right : Node or None
        Right child node (samples where feature > threshold).
    value : int, float, or None
        Class label (classification) or mean value (regression) for leaf nodes.
    impurity : float
        Impurity of the node at split time. Useful for feature importance.
    n_samples : int
        Number of training samples that reached this node.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 *, value=None, impurity=0.0, n_samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples

    def is_leaf_node(self) -> bool:
        """Return True if this node is a leaf."""
        return self.value is not None


class DecisionTree:
    """
    Decision Tree classifier and regressor using recursive binary splitting.

    Supports both classification (entropy or Gini impurity) and regression
    (variance reduction). The tree is grown greedily by selecting the split
    that maximizes information gain at each node.

    Parameters
    ----------
    min_samples_split : int, optional (default=2)
        Minimum number of samples required to split an internal node.
    max_depth : int, optional (default=100)
        Maximum depth of the tree. Controls overfitting.
    n_features : int or None, optional (default=None)
        Number of features to consider at each split. If None, all features
        are used. Setting this to sqrt(n_features) mimics Random Forest behavior.
    criterion : str, optional (default='entropy')
        Impurity measure for classification splits.
        One of ``'entropy'`` or ``'gini'``.
    task : str, optional (default='classification')
        Whether to perform ``'classification'`` or ``'regression'``.
    random_state : int or None, optional (default=None)
        Seed for reproducible feature sampling.

    Attributes
    ----------
    root : Node
        Root node of the fitted tree.
    feature_importances_ : np.ndarray of shape (n_features,)
        Normalized importance of each feature based on impurity reduction.
        Only available after fitting.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from mlpackage.supervised_learning.decision_tree import DecisionTree
    >>> import numpy as np
    >>> X = np.array([[0,0],[1,1],[0,1],[1,0]], dtype=float)
    >>> y = np.array([0, 1, 0, 1])
    >>> clf = DecisionTree(max_depth=3)
    >>> clf.fit(X, y)
    >>> clf.predict(X)
    array([0, 1, 0, 1])
    """

    def __init__(self, min_samples_split: int = 2, max_depth: int = 100,
                 n_features: int = None, criterion: str = "entropy",
                 task: str = "classification", random_state: int = None):
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if criterion not in ("entropy", "gini"):
            raise ValueError(f"criterion must be 'entropy' or 'gini', got '{criterion}'")
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.task = task
        self.random_state = random_state
        self.root: Node = None
        self.feature_importances_: np.ndarray = None
        self.n_features_in_: int = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """
        Build the decision tree from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target labels (classification) or values (regression).

        Returns
        -------
        self : DecisionTree
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

        self.n_features_in_ = X.shape[1]
        self._n_features_split = (
            self.n_features if self.n_features
            else self.n_features_in_
        )
        self._n_features_split = min(self._n_features_split, self.n_features_in_)
        self._importances = np.zeros(self.n_features_in_)

        self.root = self._grow_tree(X, y, depth=0)

        total = self._importances.sum()
        self.feature_importances_ = (
            self._importances / total if total > 0 else self._importances
        )
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively grow the decision tree.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix at the current node.
        y : np.ndarray
            Target array at the current node.
        depth : int
            Current depth in the tree.

        Returns
        -------
        Node
            A leaf node or an internal split node.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            return Node(value=self._leaf_value(y),
                        impurity=self._impurity(y),
                        n_samples=n_samples)

        feat_idxs = self._rng.choice(n_feats, self._n_features_split, replace=False)
        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)

        # If no informative split found, make a leaf
        if best_gain <= 0:
            return Node(value=self._leaf_value(y),
                        impurity=self._impurity(y),
                        n_samples=n_samples)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left  = self._grow_tree(X[left_idxs],  y[left_idxs],  depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        # Accumulate feature importance (weighted impurity reduction)
        self._importances[best_feat] += best_gain * n_samples

        return Node(best_feat, best_thresh, left, right,
                    impurity=self._impurity(y), n_samples=n_samples)

    def _best_split(self, X: np.ndarray, y: np.ndarray,
                    feat_idxs: np.ndarray) -> tuple:
        """
        Find the feature and threshold that maximise information gain.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix at the current node.
        y : np.ndarray
            Target array at the current node.
        feat_idxs : np.ndarray
            Indices of features to consider.

        Returns
        -------
        best_feat : int
            Index of the best feature.
        best_thresh : float
            Threshold value for the best split.
        best_gain : float
            Information gain of the best split.
        """
        best_gain = -np.inf
        best_feat, best_thresh = None, None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)
            for thresh in thresholds:
                gain = self._information_gain(y, X_col, thresh)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _information_gain(self, y: np.ndarray, X_col: np.ndarray,
                          thresh: float) -> float:
        """
        Compute information gain for a candidate split.

        Parameters
        ----------
        y : np.ndarray
            Target array.
        X_col : np.ndarray
            Single feature column.
        thresh : float
            Candidate split threshold.

        Returns
        -------
        float
            Information gain (parent impurity minus weighted child impurity).
        """
        left_idxs, right_idxs = self._split(X_col, thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        parent_imp = self._impurity(y)
        child_imp = (n_l / n) * self._impurity(y[left_idxs]) + \
                    (n_r / n) * self._impurity(y[right_idxs])
        return parent_imp - child_imp

    def _impurity(self, y: np.ndarray) -> float:
        """
        Compute node impurity using the selected criterion.

        Parameters
        ----------
        y : np.ndarray
            Target array at this node.

        Returns
        -------
        float
            Impurity value (entropy, Gini, or variance).
        """
        if self.task == "regression":
            return float(np.var(y))
        if self.criterion == "entropy":
            return self._entropy(y)
        return self._gini(y)

    def _entropy(self, y: np.ndarray) -> float:
        """
        Compute Shannon entropy of a label array.

        .. math::
            H(y) = -\\sum_k p_k \\log_2 p_k
        """
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return float(-np.sum([p * np.log2(p) for p in ps if p > 0]))

    def _gini(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity of a label array.

        .. math::
            G(y) = 1 - \\sum_k p_k^2
        """
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return float(1 - np.sum(ps ** 2))

    def _split(self, X_col: np.ndarray, thresh: float) -> tuple:
        """
        Partition sample indices into left and right based on threshold.

        Parameters
        ----------
        X_col : np.ndarray
            Single feature column.
        thresh : float
            Split threshold.

        Returns
        -------
        left_idxs : np.ndarray
            Indices where X_col <= thresh.
        right_idxs : np.ndarray
            Indices where X_col > thresh.
        """
        left_idxs  = np.argwhere(X_col <= thresh).flatten()
        right_idxs = np.argwhere(X_col >  thresh).flatten()
        return left_idxs, right_idxs

    def _leaf_value(self, y: np.ndarray):
        """
        Compute the prediction value for a leaf node.

        Returns the most common label for classification,
        or the mean value for regression.
        """
        if self.task == "regression":
            return float(np.mean(y))
        return Counter(y).most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or target values for input samples.

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
        if self.root is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        X = np.array(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node):
        """
        Traverse the tree for a single sample.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Single input sample.
        node : Node
            Current node being evaluated.

        Returns
        -------
        int or float
            Predicted label or value at the leaf.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

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
import numpy as np
from collections import Counter
from .decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            # Bagging: Get a random sample of the data
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # Gather predictions from every tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to get predictions per sample: [n_trees, n_samples] -> [n_samples, n_trees]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Majority vote for each sample
        predictions = [Counter(preds).most_common(1)[0][0] for preds in tree_preds]
        return np.array(predictions)
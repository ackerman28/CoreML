import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        """
        k: Number of nearest neighbors to consider
        """
        self.k = k

    def fit(self, X, y):
        """
        KNN doesn't 'train' in the traditional sense. 
        It simply stores the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for a batch of samples.
        """
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # 1. Calculate distances from x to all points in training set
        # Using the L2 norm (Euclidean distance)
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        # 2. Get indices of the k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Get the labels associated with those indices
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
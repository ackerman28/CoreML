import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target vector.
    test_size : float
        Proportion of the dataset to include in the test split (0 to 1).
    random_state : int, optional
        Seed used by the random number generator for reproducibility.
    """
    if random_state:
        np.random.seed(random_state)
    
    # Shuffle indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    
    # Calculate split point
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Resulting score z = (x - u) / s
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute the mean and std to be used for later scaling."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """Perform standardization by centering and scaling."""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
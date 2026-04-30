import numpy as np
import copy
from collections import Counter
from typing import List, Any, Optional

def _aggregate_predictions(y_preds: np.ndarray, mode: str) -> np.ndarray:
    """Helper to merge predictions from multiple models."""
    if mode == 'hard_vote':
        def get_mode(row):
            counts = Counter(row)
            # Tie-breaker: pick the smallest label if counts are equal
            return min([k for k, v in counts.items() if v == max(counts.values())])
        return np.array([get_mode(y_preds[i, :]) for i in range(y_preds.shape[0])])
    
    elif mode == 'average':
        return np.mean(y_preds, axis=1)
    raise ValueError(f"Unknown aggregation mode: {mode}")

class _BaseBagging:
    """Base class for Bootstrap Aggregating (Bagging)."""
    def __init__(self, base_estimator: Any, n_estimators: int, random_state: Optional[int], mode: str):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.mode = mode
        self.estimators_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            # 1. Clone the template model
            clf = copy.deepcopy(self.base_estimator)
            
            # 2. Create Bootstrap Sample (N samples with replacement)
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            
            # 3. Train on the sample
            clf.fit(X[indices], y[indices])
            self.estimators_.append(clf)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Collect predictions into a matrix (n_samples, n_estimators)
        all_preds = np.array([est.predict(X) for est in self.estimators_]).T
        return _aggregate_predictions(all_preds, self.mode)
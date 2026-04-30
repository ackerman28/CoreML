# Ensemble Methods: _BaseBagging Engine

This module implements **Bootstrap Aggregating (Bagging)**, a powerful ensemble technique designed to reduce variance and prevent overfitting.

## How it Works
The `_BaseBagging` class acts as a wrapper for any base estimator in the library. It follows a three-step process:

1. **Bootstrapping**: For each of the $N$ estimators, it creates a random subset of the training data by sampling with replacement.
2. **Parallel Training**: Each estimator is trained independently on its respective bootstrap sample.
3. **Aggregation**: 
   - **Classification**: Uses **Hard Voting** (majority rule) to determine the final class.
   - **Regression**: Uses **Averaging** to determine the final value.

## Why use _BaseBagging?
- **Stability**: By averaging multiple models, the ensemble is less sensitive to noise in the training data.
- **Versatility**: Unlike a standard Random Forest, this engine can "bag" any algorithm, including KNN or Perceptrons.
- **Simplicity**: High-level logic is decoupled from the specific implementation of the base learners.

## Usage Example
```python
from mlpackage.supervised_learning import _BaseBagging, DecisionTree

# Define the ensemble
forest = _BaseBagging(
    base_estimator=DecisionTree(max_depth=5),
    n_estimators=100,
    mode='hard_vote'
)

# Fit and Predict
forest.fit(X_train, y_train)
predictions = forest.predict(X_test)
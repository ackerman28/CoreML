# Decision Tree Classifier

A recursive implementation of a classification tree using Entropy and Information Gain.

## Core Features
- **Entropy-based splits**: Maximizes Information Gain at each node.
- **Recursive Structure**: Uses a custom `Node` class for tree traversal.
- **Hyperparameters**: Supports `max_depth` and `min_samples_split` for regularization.

## How it works
At each node, the algorithm iterates through all features and all unique values of those features to find the split $(j, t)$ that minimizes entropy in the resulting children.
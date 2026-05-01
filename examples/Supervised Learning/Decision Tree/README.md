# Decision Tree

A from-scratch implementation of a Decision Tree supporting classification (entropy or Gini) and regression (variance reduction), with feature importance tracking.

---

## Contents

| File | Description |
|---|---|
| `decision_tree_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/supervised_learning/decision_tree.py` | Core implementation |
| `../../tests/test_decision_tree.py` | Unit test suite |

---

## Background

A Decision Tree partitions the feature space via recursive binary splits, selecting the split that maximises information gain at each node:

$$\text{Information Gain} = H(\text{parent}) - \left[\frac{n_L}{n} H(\text{left}) + \frac{n_R}{n} H(\text{right})\right]$$

Two impurity measures are supported for classification:

| Criterion | Formula |
|---|---|
| **Entropy** | $H(y) = -\sum_k p_k \log_2 p_k$ |
| **Gini** | $G(y) = 1 - \sum_k p_k^2$ |

For regression, **variance reduction** is used as the splitting criterion.

Feature importance is computed as the total weighted impurity reduction across all splits on each feature, normalized to sum to 1.

---

## Experiments

### 1 — Effect of max_depth
Visualizes decision boundaries on the moons dataset for depths ∈ {1, 2, 5, 10}, illustrating the bias-variance tradeoff.

### 2 — Entropy vs Gini
Side-by-side comparison of both criteria on the same dataset, demonstrating that they typically produce equivalent boundaries.

### 3 — Overfitting Analysis
Sweeps `max_depth` from 1 to 19 and plots train vs test accuracy, revealing the optimal depth and the onset of overfitting.

### 4 — Iris Dataset & Feature Importances
Applies the tree to the Iris dataset (4 features, 3 classes). Reports accuracy, classification report, confusion matrix, and a feature importance bar chart.

### 5 — Breast Cancer Dataset
Applies the tree to the Breast Cancer dataset (30 features). Reports accuracy, classification report, confusion matrix, and top-10 feature importances.

---

## Results Summary

| Experiment | Result |
|---|---|
| Moons (best depth) | ~90% test accuracy |
| Iris (depth=5) | ~97% test accuracy |
| Breast Cancer (depth=5) | ~93% test accuracy |

---

## Usage

```python
from src.mlpackage.supervised_learning.decision_tree import DecisionTree

# Classification
clf = DecisionTree(max_depth=5, criterion="entropy", random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.feature_importances_)

# Regression
reg = DecisionTree(task="regression", max_depth=5)
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `100` | Maximum tree depth — primary regularization knob |
| `min_samples_split` | `2` | Minimum samples required to split a node |
| `criterion` | `'entropy'` | Impurity measure: `'entropy'` or `'gini'` |
| `task` | `'classification'` | `'classification'` or `'regression'` |
| `n_features` | `None` | Features considered per split (None = all) |
| `random_state` | `None` | Seed for reproducible feature sampling |

---

## Notes

- Decision Trees do **not** require feature standardization — splits are threshold-based and scale-invariant.
- `max_depth` is the most important hyperparameter. Use cross-validation to find the optimal value.
- Setting `n_features` to `int(sqrt(n_features))` enables Random Forest-style feature subsampling.
- `feature_importances_` is available after fitting and reflects the relative contribution of each feature to impurity reduction across the full tree.
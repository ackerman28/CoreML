# Random Forest

A from-scratch Random Forest implementation supporting classification and regression, with OOB scoring, feature importances, and probability estimation.

---

## Contents

| File | Description |
|---|---|
| `random_forest_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/supervised_learning/random_forest.py` | Core implementation |
| `../../tests/test_random_forest.py` | Unit test suite |

---

## Background

Random Forest builds $B$ Decision Trees, each on a bootstrap sample, and aggregates their predictions:

**Classification (majority vote):**
$$\hat{y} = \text{mode}\lbrace h_1(\mathbf{x}), \dots, h_B(\mathbf{x}) \rbrace$$

**Regression (mean):**
$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} h_b(\mathbf{x})$$

Two randomization mechanisms reduce variance and decorrelate trees: **bootstrap sampling** of rows and **feature subsampling** ($\sqrt{p}$ features per split) of columns.

The **OOB score** uses the ~36.8% of samples excluded from each tree's bootstrap sample as a free validation estimate.

---

## Experiments

### 1 — Random Forest vs Decision Tree
Side-by-side boundary comparison on the moons dataset, showing how the ensemble produces a smoother, more robust boundary than a single tree.

### 2 — Effect of n_trees
Plots train accuracy, test accuracy, and OOB score as the number of trees grows from 1 to 200 — showing variance reduction with more trees.

### 3 — Feature Importances (Iris)
Bar chart of averaged feature importances on the Iris dataset alongside a confusion matrix.

### 4 — Breast Cancer Dataset
Full evaluation on 30-feature Breast Cancer data including accuracy, OOB score, classification report, confusion matrix, and top-10 feature importances.

---

## Results Summary

| Experiment | Result |
|---|---|
| Moons (100 trees) | ~90% test accuracy |
| Iris (100 trees) | ~97% test accuracy |
| Breast Cancer (100 trees) | ~96% test accuracy |

---

## Usage

```python
from src.mlpackage.supervised_learning.random_forest import RandomForest

rf = RandomForest(n_trees=100, max_depth=8, random_state=42)
rf.fit(X_train, y_train, oob_score=True)

print(rf.score(X_test, y_test))
print(rf.oob_score_)
print(rf.feature_importances_)
print(rf.predict_proba(X_test))
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `n_trees` | `100` | Number of trees in the forest |
| `max_depth` | `10` | Maximum depth of each tree |
| `criterion` | `'entropy'` | Split criterion: `'entropy'` or `'gini'` |
| `task` | `'classification'` | `'classification'` or `'regression'` |
| `n_features` | `None` | Features per split (None = sqrt(p)) |
| `random_state` | `None` | Seed for reproducibility |

---

## Notes

- Random Forest does **not** require feature standardization.
- Use `oob_score=True` to get a free generalization estimate without a validation set.
- More trees always help (up to diminishing returns around 100–200 for most datasets).
- `feature_importances_` is averaged across all trees, making it more stable than a single tree's importances.
# Ensemble Methods

From-scratch ensemble learning: Bagging (with classification and regression variants) and hard/soft Voting over diverse estimators.

---

## Contents

| File | Description |
|---|---|
| `ensemble_analysis.ipynb` | End-to-end analysis comparing ensemble strategies |
| `../../src/mlpackage/supervised_learning/ensemble.py` | Core implementation |
| `../../tests/test_ensemble.py` | Unit test suite |

---

## Background

Ensemble methods reduce the **bias-variance tradeoff** by combining multiple models:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

**Bagging** reduces variance by averaging over $B$ models trained on bootstrap samples:

$$\hat{y} = \text{mode}\{h_1(\mathbf{x}), \dots, h_B(\mathbf{x})\} \quad \text{(classification)}$$
$$\hat{y} = \frac{1}{B}\sum_{b=1}^B h_b(\mathbf{x}) \quad \text{(regression)}$$

**Voting** combines diverse model types — leveraging their complementary strengths.

---

## Implemented Classes

| Class | Strategy | Description |
|---|---|---|
| `_BaseBagging` | Bagging | Generic bootstrap aggregating |
| `BaggingClassifier` | Bagging | Hard-vote + `predict_proba()` |
| `BaggingRegressor` | Bagging | Mean-averaging for regression |
| `VotingClassifier` | Voting | Hard or soft voting over named estimators |

---

## Experiments

### 1 — Single Tree vs Bagging
Side-by-side decision boundary comparison on the moons dataset — showing how bagging produces a smoother, more robust boundary.

### 2 — Effect of n_estimators
Plots train and test accuracy as the ensemble size grows from 1 to 100 estimators.

### 3 — Voting Classifier Comparison
Compares Decision Tree, KNN, Logistic Regression, and a Voting ensemble on the Breast Cancer dataset.

### 4 — Bagging Full Evaluation
Complete evaluation of BaggingClassifier on Breast Cancer including accuracy, classification report, and confusion matrix.

---

## Usage

```python
from src.mlpackage.supervised_learning.ensemble import (
    BaggingClassifier, BaggingRegressor, VotingClassifier
)
from src.mlpackage.supervised_learning.decision_tree import DecisionTree
from src.mlpackage.supervised_learning.knn import KNN

# Bagging
clf = BaggingClassifier(DecisionTree(max_depth=5), n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.predict_proba(X_test))

# Voting
voter = VotingClassifier([("dt", DecisionTree()), ("knn", KNN())], voting="hard")
voter.fit(X_train, y_train)
print(voter.score(X_test, y_test))
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `base_estimator` | required | Any object with `fit` / `predict` |
| `n_estimators` | `10` | Number of bootstrap estimators |
| `voting` | `'hard'` | `'hard'` or `'soft'` (VotingClassifier) |
| `random_state` | `None` | Seed for reproducibility |

---

## Notes

- Bagging is most effective with **high-variance base learners** like deep decision trees.
- For voting to help, the component models should make **different kinds of errors** — diversity is the key ingredient.
- Soft voting generally outperforms hard voting when the models produce well-calibrated probabilities.
# Logistic Regression

A from-scratch implementation of Logistic Regression for binary classification, trained via batch gradient descent with binary cross-entropy loss.

---

## Contents

| File | Description |
|---|---|
| `logistic_regression_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/supervised_learning/logistic_regression.py` | Core implementation |
| `../../tests/test_logistic_regression.py` | Unit test suite |

---

## Background

Logistic Regression estimates the probability of class membership using the sigmoid function:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$

Parameters are learned by minimizing **binary cross-entropy loss**:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i) \right]$$

Gradient descent update rules:

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{n} X^\top(\hat{p} - y), \qquad b \leftarrow b - \frac{\eta}{n}\sum(\hat{p}_i - y_i)$$

---

## Experiments

### 1 — Decision Boundary (Synthetic)
Visualizes the linear decision boundary learned on a 2D synthetic dataset and reports accuracy.

### 2 — Loss Convergence
Tracks binary cross-entropy over 300 iterations across four learning rates to illustrate the effect of step size on convergence speed and stability.

### 3 — Decision Threshold Analysis
Sweeps the decision threshold from 0.1 to 0.9 and plots how accuracy, precision, and recall change — useful for understanding the precision-recall tradeoff.

### 4 — Breast Cancer Dataset
Applies the model to the UCI Breast Cancer dataset (30 features, 569 samples). Evaluates with accuracy, full classification report, and confusion matrix.

### 5 — ROC Curve & AUC
Plots the Receiver Operating Characteristic curve and reports AUC — a threshold-independent measure of classifier quality.

---

## Results Summary

| Experiment | Result |
|---|---|
| Synthetic 2D (train) | ~95% accuracy |
| Breast Cancer (test) | ~95% accuracy |
| AUC (Breast Cancer) | ~0.99 |

---

## Usage

```python
from src.mlpackage import LogisticRegression
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

model = LogisticRegression(learning_rate=0.1, n_iters=500, threshold=0.5)
model.fit(X_train_s, y_train)

print(model.score(X_test_s, y_test))
print(model.predict_proba(X_test_s))
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `0.01` | Gradient descent step size |
| `n_iters` | `1000` | Number of training iterations |
| `threshold` | `0.5` | Decision threshold for class assignment |

---

## Notes

- Always **standardize features** before training — logistic regression is sensitive to feature scale.
- Use `predict_proba()` when you need calibrated probabilities rather than hard labels.
- Adjust `threshold` to tune the precision-recall tradeoff based on your application (e.g. lower threshold for higher recall in medical screening).
- `loss_history` tracks binary cross-entropy per iteration — a flat loss curve suggests the learning rate is too small or the model has converged.
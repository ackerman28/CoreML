# Perceptron

A from-scratch implementation of Rosenblatt's Perceptron — the foundational binary linear classifier and precursor to modern neural networks.

---

## Contents

| File | Description |
|---|---|
| `perceptron_analysis.ipynb` | End-to-end analysis across synthetic and real datasets |
| `../../src/mlpackage/supervised_learning/perceptron.py` | Core implementation |
| `../../tests/test_perceptron.py` | Unit test suite |

---

## Background

The Perceptron learns a linear decision boundary by iterating over training samples and correcting its weights whenever a prediction is wrong. Given input **x**, it predicts:

$$\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w} \cdot \mathbf{x} + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

On each misclassification, weights are updated via:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y_i - \hat{y}_i)\,\mathbf{x}_i, \qquad b \leftarrow b + \eta\,(y_i - \hat{y}_i)$$

Convergence is **guaranteed** when data is linearly separable (Perceptron Convergence Theorem). On non-separable data (e.g. XOR), it will not converge.

---

## Experiments

### 1 — Linearly Separable vs XOR
Demonstrates convergence on separable data and the fundamental limitation on non-linearly separable problems.

### 2 — Convergence Across Learning Rates
Tracks misclassifications per epoch for different learning rates (`0.001` → `0.5`) to illustrate the effect of step size on training stability.

### 3 — Breast Cancer Dataset
Applies the Perceptron to the UCI Breast Cancer dataset (30 features, 569 samples) after standardization. Evaluates with accuracy, precision, recall, and a confusion matrix.

### 4 — Accuracy vs Iterations
Shows how train/test accuracy evolves as the number of training epochs increases, revealing the point of diminishing returns.

---

## Results Summary

| Experiment | Result |
|---|---|
| Linearly separable (synthetic) | 100% accuracy |
| XOR | < 100% (not linearly separable) |
| Breast Cancer (test set) | ~95% accuracy |

---

## Usage

```python
from src.mlpackage import Perceptron
import numpy as np

X_train, X_test, y_train, y_test = ...   # your data, standardized

clf = Perceptron(learning_rate=0.1, n_iters=300)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
print(clf.predict(X_test))
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `0.01` | Step size for weight updates |
| `n_iters` | `1000` | Number of passes over training data |

---

## Notes

- Features should be **standardized** before training (zero mean, unit variance). The Perceptron is sensitive to feature scale.
- Labels can be any binary values — they are internally mapped to `{0, 1}`.
- `training_errors` attribute tracks misclassifications per epoch, useful for diagnosing convergence.
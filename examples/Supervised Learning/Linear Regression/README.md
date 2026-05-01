# Linear Regression

A from-scratch implementation of Linear Regression supporting both the closed-form Normal Equation and iterative Gradient Descent solvers.

---

## Contents

| File | Description |
|---|---|
| `linear_regression_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/supervised_learning/linear_regression.py` | Core implementation |
| `../../tests/test_linear_regression.py` | Unit test suite |

---

## Background

Linear Regression models the relationship between features and a continuous target as:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

Parameters are chosen to minimize Mean Squared Error:

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Two solvers are implemented:

**Normal Equation** — exact closed-form solution:

$$\boldsymbol{\theta} = (X^\top X)^{-1} X^\top \mathbf{y}$$

**Gradient Descent** — iterative update rule:

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{2\eta}{n} X^\top(\hat{y} - y), \qquad b \leftarrow b - \frac{2\eta}{n}\sum(\hat{y} - y)$$

---

## Experiments

### 1 — 1D Regression Line
Fits a line to synthetic noisy data and verifies the recovered slope and intercept against the known ground truth.

### 2 — Normal Equation vs Gradient Descent
Side-by-side comparison of both solvers on the same dataset, confirming they converge to equivalent solutions.

### 3 — Loss Convergence
Tracks MSE over gradient descent iterations across learning rates (`0.001` → `0.1`) to illustrate convergence behavior and the effect of step size.

### 4 — California Housing Dataset
Applies the model to a real-world dataset (20,640 samples) using median income as a single predictor of house value. Evaluates with R², MSE, and RMSE.

### 5 — Residual Analysis
Plots residuals vs predicted values and their distribution to validate the homoscedasticity assumption and check for systematic bias.

---

## Results Summary

| Experiment | Result |
|---|---|
| 1D synthetic (noise-free) | R² ≈ 1.00 |
| California Housing (test) | R² ≈ 0.47 (single feature) |
| Both solvers | Produce equivalent R² |

---

## Usage

```python
from src.mlpackage import LinearRegression
import numpy as np

# Normal Equation
model = LinearRegression(method="normal")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# Gradient Descent
model = LinearRegression(method="gradient_descent", learning_rate=0.05, n_iters=500)
model.fit(X_train, y_train)
print(model.loss_history[-1])
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `method` | `'normal'` | Solver: `'normal'` or `'gradient_descent'` |
| `learning_rate` | `0.01` | Step size for gradient descent |
| `n_iters` | `1000` | Number of gradient descent iterations |

---

## Notes

- Always **standardize features** before fitting, especially with gradient descent.
- The Normal Equation uses `np.linalg.pinv` for numerical stability with collinear features.
- `loss_history` is populated only when using gradient descent — useful for diagnosing convergence.
- R² can be negative if the model is worse than predicting the mean (typically a sign of wrong hyperparameters or unscaled features).
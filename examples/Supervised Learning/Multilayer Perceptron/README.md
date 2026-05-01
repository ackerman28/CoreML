# Multilayer Perceptron (MLP)

A from-scratch implementation of a fully-connected feedforward neural network trained via backpropagation. Supports non-linear classification tasks that single-layer models cannot solve.

---

## Contents

| File | Description |
|---|---|
| `mlp_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/supervised_learning/multilayer_perceptron.py` | Core implementation |
| `../../tests/test_multilayer_perceptron.py` | Unit test suite |

---

## Background

The MLP extends the Perceptron by introducing one or more hidden layers with non-linear activations. The forward pass computes:

$$\mathbf{a}^{(1)} = \sigma(X W^{(1)} + \mathbf{b}^{(1)}), \qquad \mathbf{a}^{(2)} = \sigma(\mathbf{a}^{(1)} W^{(2)} + \mathbf{b}^{(2)})$$

Weights are learned by backpropagating the MSE error signal through the network using the chain rule:

$$\boldsymbol{\delta}^{(2)} = (y - \mathbf{a}^{(2)}) \odot \sigma'(\mathbf{a}^{(2)}), \qquad \boldsymbol{\delta}^{(1)} = (\boldsymbol{\delta}^{(2)} {W^{(2)}}^\top) \odot \sigma'(\mathbf{a}^{(1)})$$

Weights are initialized with **Xavier normal initialization** to keep activations in a healthy range at the start of training.

---

## Experiments

### 1 — XOR Problem
Demonstrates that the MLP solves XOR — a problem that is provably unsolvable by a single-layer Perceptron. Shows the non-linear decision boundary and the loss curve over 20,000 epochs.

### 2 — Effect of Hidden Size (Moons)
Trains four models with hidden sizes `[2, 4, 8, 16]` on the moons dataset, visualizing how network capacity affects the complexity of the learned boundary.

### 3 — Loss Convergence
Tracks MSE over 500 epochs across four learning rates (`0.01` → `0.5`) to illustrate convergence behavior and sensitivity to step size.

### 4 — Breast Cancer Dataset
Applies the MLP (30 inputs, 16 hidden neurons) to the UCI Breast Cancer dataset. Reports accuracy, classification report, confusion matrix, and training loss curve.

---

## Results Summary

| Experiment | Result |
|---|---|
| XOR | 100% accuracy |
| Moons (hidden=16) | ~98% accuracy |
| Breast Cancer (test) | ~97% accuracy |

---

## Usage

```python
from src.mlpackage import MultilayerPerceptron
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

model = MultilayerPerceptron(
    input_size=X_train.shape[1],
    hidden_size=16,
    output_size=1,
    learning_rate=0.1,
    n_iters=1000
)
model.fit(X_train_s, y_train)

print(model.score(X_test_s, y_test))
print(model.predict_proba(X_test_s))
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `input_size` | required | Number of input features |
| `hidden_size` | required | Number of hidden neurons |
| `output_size` | required | Number of output neurons |
| `learning_rate` | `0.1` | Gradient descent step size |
| `n_iters` | `1000` | Number of training epochs |
| `random_state` | `42` | Seed for reproducible initialization |

---

## Notes

- Always **standardize features** before training. The sigmoid activation is sensitive to input scale.
- Increase `hidden_size` or `n_iters` for more complex datasets.
- `loss_history` tracks MSE per epoch — useful for diagnosing convergence and choosing learning rate.
- Xavier initialization is used by default, which significantly improves convergence compared to uniform random initialization.
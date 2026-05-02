# CoreML

A lightweight machine learning library built entirely from scratch using NumPy — implementing fundamental supervised and unsupervised algorithms from first principles.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-only-orange)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

CoreML prioritizes **mathematical transparency** over abstraction. Every algorithm is implemented from the ground up — no scikit-learn, no PyTorch — just linear algebra and NumPy. The goal is to make the mechanics of machine learning visible and understandable.

Each implementation includes:
- Clean, documented source code with full docstrings
- A Jupyter notebook with multiple experiments and real dataset analysis
- A comprehensive unit test suite (pytest)
- A dedicated README with mathematical background

---

## Algorithms

### Supervised Learning

| Algorithm | Description | Notebook |
|---|---|---|
| [Perceptron](examples/Supervised%20Learning/Perceptron/) | Binary linear classifier via the Perceptron Learning Rule | [📓](examples/Supervised%20Learning/Perceptron/perceptron_analysis.ipynb) |
| [Linear Regression](examples/Supervised%20Learning/Linear%20Regression/) | Normal equation and gradient descent solvers | [📓](examples/Supervised%20Learning/Linear%20Regression/linear_regression_analysis.ipynb) |
| [Logistic Regression](examples/Supervised%20Learning/Logistic%20Regression/) | Binary classifier with sigmoid + cross-entropy loss | [📓](examples/Supervised%20Learning/Logistic%20Regression/logistic_regression_analysis.ipynb) |
| [Multilayer Perceptron](examples/Supervised%20Learning/Neural%20Networks/) | Feedforward neural network trained via backpropagation | [📓](examples/Supervised%20Learning/Neural%20Networks/mlp_analysis.ipynb) |
| [K-Nearest Neighbors](examples/Supervised%20Learning/KNN/) | Instance-based classifier and regressor | [📓](examples/Supervised%20Learning/KNN/knn_analysis.ipynb) |
| [Decision Tree](examples/Supervised%20Learning/Decision%20Trees/) | CART with entropy/Gini, regression, and feature importances | [📓](examples/Supervised%20Learning/Decision%20Trees/decision_tree_analysis.ipynb) |
| [Random Forest](examples/Supervised%20Learning/Random%20Forest/) | Bagging ensemble of decision trees with OOB scoring | [📓](examples/Supervised%20Learning/Random%20Forest/random_forest_analysis.ipynb) |
| [Ensemble Methods](examples/Supervised%20Learning/Ensembles/) | Bagging classifier/regressor and Voting classifier | [📓](examples/Supervised%20Learning/Ensembles/ensemble_analysis.ipynb) |

### Unsupervised Learning

| Algorithm | Description | Notebook |
|---|---|---|
| [PCA](examples/Unsupervised%20Learning/PCA/) | Eigendecomposition-based dimensionality reduction | [📓](examples/Unsupervised%20Learning/PCA/pca_analysis.ipynb) |
| [K-Means](examples/Unsupervised%20Learning/KMeans/) | Lloyd's algorithm with K-Means++ initialization | [📓](examples/Unsupervised%20Learning/KMeans/kmeans_analysis.ipynb) |
| [DBSCAN](examples/Unsupervised%20Learning/DBSCAN/) | Density-based clustering with noise detection | [📓](examples/Unsupervised%20Learning/DBSCAN/dbscan_analysis.ipynb) |

---

## Project Structure

```text
CoreML/
├── examples/
│   ├── Supervised Learning/
│   │   ├── Decision Trees/
│   │   │   ├── decision_tree_analysis.ipynb
│   │   │   └── README.md
│   │   ├── Ensembles/
│   │   │   ├── ensemble_analysis.ipynb
│   │   │   └── README.md
│   │   ├── KNN/
│   │   │   ├── knn_analysis.ipynb
│   │   │   └── README.md
│   │   ├── Linear Regression/
│   │   │   ├── linear_regression_analysis.ipynb
│   │   │   └── README.md
│   │   ├── Logistic Regression/
│   │   │   ├── logistic_regression_analysis.ipynb
│   │   │   └── README.md
│   │   ├── Neural Networks/
│   │   │   ├── mlp_analysis.ipynb
│   │   │   └── README.md
│   │   ├── Perceptron/
│   │   │   ├── perceptron_analysis.ipynb
│   │   │   └── README.md
│   │   └── Random Forest/
│   │       ├── random_forest_analysis.ipynb
│   │       └── README.md
│   └── Unsupervised Learning/
│       ├── DBSCAN/
│       │   ├── dbscan_analysis.ipynb
│       │   └── README.md
│       ├── KMeans/
│       │   ├── kmeans_analysis.ipynb
│       │   └── README.md
│       └── PCA/
│           ├── pca_analysis.ipynb
│           └── README.md
├── src/
│   └── mlpackage/
│       ├── __init__.py
│       ├── metrics.py
│       ├── preprocess.py
│       ├── supervised_learning/
│       │   ├── __init__.py
│       │   ├── perceptron.py
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── multilayer_perceptron.py
│       │   ├── knn.py
│       │   ├── decision_tree.py
│       │   ├── random_forest.py
│       │   └── ensemble.py
│       └── unsupervised_learning/
│           ├── __init__.py
│           ├── pca.py
│           ├── kmeans.py
│           └── dbscan.py
├── tests/
│   ├── test_perceptron.py
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_multilayer_perceptron.py
│   ├── test_knn.py
│   ├── test_decision_tree.py
│   ├── test_random_forest.py
│   ├── test_ensemble.py
│   ├── test_pca.py
│   ├── test_kmeans.py
│   └── test_dbscan.py
├── setup.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ackerman28/CoreML.git
cd CoreML
pip install -e .
```

**Requirements:** Python 3.8+, NumPy, scikit-learn (examples only), matplotlib (examples only).

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from mlpackage.supervised_learning.random_forest import RandomForest
from mlpackage.supervised_learning.logistic_regression import LogisticRegression
from mlpackage.unsupervised_learning.kmeans import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# Supervised — Random Forest
rf = RandomForest(n_trees=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest  accuracy: {rf.score(X_test, y_test):.2%}")

# Supervised — Logistic Regression
lr = LogisticRegression(learning_rate=0.1, n_iters=500)
lr.fit(X_train, y_train)
print(f"Logistic Regression accuracy: {lr.score(X_test, y_test):.2%}")

# Unsupervised — K-Means
km = KMeans(k=2, random_state=42)
km.fit(X_train)
print(f"K-Means inertia: {km.inertia_:.2f}")
```

---

## Testing

All algorithms are covered by a pytest test suite. Tests are automatically triggered on every pull request via GitHub Actions CI.

```bash
# Run all tests
pytest tests/

# Run tests for a specific algorithm
pytest tests/test_random_forest.py -v

# Run with coverage
pytest tests/ --cov=src/mlpackage
```

---

## Design Philosophy

- **From scratch** — every algorithm uses only NumPy. No ML framework dependencies.
- **Mathematically grounded** — implementations reflect the underlying linear algebra, not just API calls.
- **Consistent interface** — all estimators follow a `fit()` / `predict()` / `score()` pattern.
- **Readable** — code is written to be understood, not just to run fast.
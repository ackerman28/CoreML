# K-Nearest Neighbors (KNN)

A from-scratch implementation of K-Nearest Neighbors supporting classification and regression, multiple distance metrics, and distance-based neighbor weighting.

---

## Contents

| File | Description |
|---|---|
| `knn_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/supervised_learning/knn.py` | Core implementation |
| `../../tests/test_knn.py` | Unit test suite | 

---

## Background

KNN is a **non-parametric, lazy learning** algorithm. It stores the training set and makes predictions by finding the K closest points to a query:

$$\hat{y} = \text{majority}\left\{ y^{(i)} : i \in \mathcal{N}_k(\mathbf{x}) \right\} \quad \text{(classification)}$$

$$\hat{y} = \frac{1}{K}\sum_{i \in \mathcal{N}_k(\mathbf{x})} y^{(i)} \quad \text{(regression)}$$

Three distance metrics are supported:

| Metric | Formula |
|---|---|
| Euclidean | $\sqrt{\sum_j (x_j - z_j)^2}$ |
| Manhattan | $\sum_j \|x_j - z_j\|$ |
| Minkowski | $\left(\sum_j \|x_j - z_j\|^p\right)^{1/p}$ |

With distance weighting, neighbor contributions are scaled by $w_i = 1/d_i$.

---

## Experiments

### 1 — Effect of K on Decision Boundary
Visualizes decision boundaries on the moons dataset for K ∈ {1, 5, 15, 30}, demonstrating how K controls the bias-variance tradeoff.

### 2 — Optimal K Selection
Sweeps K from 1 to 30 on the moons dataset and plots train vs test accuracy to identify the optimal K.

### 3 — Distance Metric Comparison
Side-by-side comparison of Euclidean, Manhattan, and Minkowski boundaries on the same dataset.

### 4 — Iris Dataset (Multiclass)
Applies KNN (K=5) to the Iris dataset (4 features, 3 classes). Reports accuracy, classification report, and confusion matrix.

### 5 — Uniform vs Distance Weighting
Compares uniform and distance-weighted KNN across K=1..20 on the Breast Cancer dataset.

---

## Results Summary

| Experiment | Result |
|---|---|
| Moons (best K) | ~90% test accuracy |
| Iris (K=5) | ~97% test accuracy |
| Breast Cancer (best K) | ~96% test accuracy |

---

## Usage

```python
from src.mlpackage.supervised_learning.knn import KNN
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

# Classification
clf = KNN(k=5, metric="euclidean", weights="distance")
clf.fit(X_train_s, y_train)
print(clf.score(X_test_s, y_test))

# Regression
reg = KNN(k=5, task="regression")
reg.fit(X_train_s, y_train)
print(reg.predict(X_test_s))
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `k` | `3` | Number of nearest neighbors |
| `metric` | `'euclidean'` | Distance metric: `'euclidean'`, `'manhattan'`, `'minkowski'` |
| `p` | `2` | Power for Minkowski metric |
| `task` | `'classification'` | `'classification'` or `'regression'` |
| `weights` | `'uniform'` | `'uniform'` or `'distance'` |

---

## Notes

- Always **standardize features** before using KNN. Distance metrics are sensitive to feature scale — a feature with large magnitude will dominate the distance computation.
- KNN has **no training cost** but prediction is O(n·d) per query, which becomes slow for large datasets.
- Use **odd K** for binary classification to avoid ties.
- Distance weighting generally helps near class boundaries where uniform voting may be ambiguous.
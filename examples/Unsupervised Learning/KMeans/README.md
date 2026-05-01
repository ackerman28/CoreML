# K-Means Clustering

A from-scratch K-Means implementation with K-Means++ initialization, multiple restarts, and inertia-based model selection.

---

## Contents

| File | Description |
|---|---|
| `kmeans_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/unsupervised_learning/kmeans.py` | Core implementation |
| `../../tests/test_kmeans.py` | Unit test suite |

---

## Background

K-Means minimizes **within-cluster sum of squares (WCSS / inertia)**:

$$\mathcal{L} = \sum_{j=1}^{k} \sum_{i \in C_j} \|\mathbf{x}_i - \mu_j\|^2$$

The algorithm alternates between two steps until convergence:

**Assignment:** $c_i = \arg\min_j \|\mathbf{x}_i - \mu_j\|^2$

**Update:** $\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} \mathbf{x}_i$

**K-Means++** initialization seeds centroids with probability proportional to $d(\mathbf{x}, \mathcal{C})^2$, reducing the risk of poor convergence.

---

## Experiments

### 1 — K-Means on Synthetic Blobs
Visualizes ground truth vs K-Means labels on 4-cluster synthetic data. Reports Adjusted Rand Index (ARI) and inertia.

### 2 — Elbow Method
Plots inertia and silhouette score across k=1..10 to demonstrate how to select the optimal number of clusters.

### 3 — KMeans++ vs Random Initialization
Runs 20 independent trials of each initialization strategy and compares inertia distributions, demonstrating the reliability advantage of K-Means++.

### 4 — Iris Dataset
Applies K-Means (k=3) to the Iris dataset, projecting to 2D with PCA for visualization. Reports ARI and silhouette score.

### 5 — Limitation: Non-Spherical Clusters
Demonstrates that K-Means fails on the moons dataset, motivating density-based alternatives like DBSCAN.

---

## Results Summary

| Experiment | Result |
|---|---|
| Synthetic blobs (k=4) | ARI ≈ 1.0 |
| Iris (k=3) | ARI ≈ 0.73 |
| Moons | Poor — not spherical |

---

## Usage

```python
from src.mlpackage.unsupervised_learning.kmeans import KMeans
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

km = KMeans(k=3, init="kmeans++", n_init=10, random_state=42)
km.fit(X_scaled)

print(km.labels_)
print(km.inertia_)
print(km.centroids_)
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `k` | `3` | Number of clusters |
| `init` | `'kmeans++'` | Initialization: `'random'` or `'kmeans++'` |
| `n_init` | `10` | Independent restarts; best inertia kept |
| `max_iters` | `300` | Maximum EM iterations per run |
| `tol` | `1e-4` | Convergence tolerance on centroid shift |
| `random_state` | `None` | Seed for reproducibility |

---

## Notes

- Always **standardize features** before K-Means. Euclidean distance is sensitive to scale.
- Use the **elbow method** and **silhouette score** together to choose `k`.
- K-Means assumes **spherical, similarly sized clusters**. For non-spherical or density-based clusters, use DBSCAN.
- `n_init=10` runs 10 independent trials by default — the best solution (lowest inertia) is retained.
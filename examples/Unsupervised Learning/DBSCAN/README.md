# DBSCAN

A from-scratch implementation of Density-Based Spatial Clustering of Applications with Noise (DBSCAN), supporting Euclidean and Manhattan distance metrics.

---

## Contents

| File | Description |
|---|---|
| `dbscan_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/unsupervised_learning/dbscan.py` | Core implementation |
| `../../tests/test_dbscan.py` | Unit test suite |

---

## Background

DBSCAN classifies each point as a **core point**, **border point**, or **noise point** based on its local neighborhood density.

A point $p$ is a **core point** if:
$$|N_\varepsilon(p)| \geq \text{min\_samples}$$

where $N_\varepsilon(p) = \{q : d(p, q) \leq \varepsilon\}$.

All density-connected points form one cluster. Points not reachable from any core point are labeled noise ($-1$).

---

## Experiments

### 1 — DBSCAN vs K-Means (Non-Spherical Data)
Compares DBSCAN and K-Means on moons and circles datasets — demonstrating that DBSCAN correctly handles non-convex cluster shapes where K-Means fails.

### 2 — K-Distance Plot
Uses the sorted k-nearest-neighbor distance curve to identify a good `eps` value — the elbow corresponds to the transition between cluster points and noise.

### 3 — Effect of eps and min_samples
Grid of decision boundaries showing how varying `eps` (0.1 → 0.8) and `min_samples` (2 → 20) affects the number of clusters and noise points detected.

### 4 — Blobs with Injected Outliers
Applies DBSCAN to a 3-cluster dataset with 20 randomly injected outliers — demonstrating robust noise detection and accurate cluster recovery (ARI ≈ 0.99).

---

## Results Summary

| Experiment | Result |
|---|---|
| Moons (DBSCAN) | 2 clusters, correct boundaries |
| Moons (K-Means) | Fails — wrong boundaries |
| Blobs + outliers | ARI ≈ 0.99, all outliers detected |

---

## Usage

```python
from src.mlpackage.unsupervised_learning.dbscan import DBSCAN
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

db = DBSCAN(eps=0.5, min_samples=5)
db.fit(X_scaled)

print(db.labels_)          # -1 = noise
print(db.n_clusters_)
print(db.n_noise_)
print(db.core_sample_indices_)
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `eps` | `0.5` | Neighborhood radius — smaller = tighter clusters |
| `min_samples` | `5` | Min points to be a core point |
| `metric` | `'euclidean'` | `'euclidean'` or `'manhattan'` |

---

## Notes

- Always **standardize features** before DBSCAN. The `eps` parameter is in the same units as feature distances.
- Use the **k-distance plot** to choose `eps` — the elbow in the sorted k-NN distances is a reliable heuristic.
- `min_samples` ≥ `n_features + 1` is a common starting point.
- DBSCAN is $O(n^2)$ in the naïve implementation — for large datasets consider indexing structures like KD-trees.
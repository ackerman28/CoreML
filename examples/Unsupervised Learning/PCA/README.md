# Principal Component Analysis (PCA)

A from-scratch implementation of PCA via eigendecomposition of the covariance matrix, with explained variance diagnostics and approximate reconstruction.

---

## Contents

| File | Description |
|---|---|
| `pca_analysis.ipynb` | End-to-end analysis on synthetic and real datasets |
| `../../src/mlpackage/unsupervised_learning/pca.py` | Core implementation |
| `../../tests/test_pca.py` | Unit test suite |

---

## Background

PCA finds the orthogonal directions of maximum variance in the data:

**1. Center:** $\tilde{X} = X - \bar{\mathbf{x}}$

**2. Covariance:** $\Sigma = \frac{1}{n-1}\tilde{X}^\top\tilde{X}$

**3. Eigendecomposition:** $\Sigma \mathbf{v}_j = \lambda_j \mathbf{v}_j$

**4. Project:** $Z = \tilde{X} V_k$

The explained variance ratio of component $j$:

$$\text{EVR}_j = \frac{\lambda_j}{\sum_i \lambda_i}$$

Data can be approximately reconstructed via:

$$\hat{X} = Z V_k^\top + \bar{\mathbf{x}}$$

---

## Experiments

### 1 — Explained Variance (Iris)
Scree plot showing individual and cumulative explained variance ratio across all 4 Iris components, alongside a 2D scatter of the projected data colored by class.

### 2 — Reconstruction Error vs Components
Plots MSE reconstruction error on the Digits dataset as the number of components grows from 2 to 64, showing the diminishing returns of adding components.

### 3 — Visual Reconstruction (Digits)
Side-by-side images of a reconstructed digit for k ∈ {2, 10, 30, 64}, visually demonstrating how reconstruction quality improves with more components.

### 4 — 2D Projection (Digits)
Projects the 64-dimensional Digits dataset to 2D and visualizes all 10 digit classes, showing natural clustering in the principal component space.

---

## Results Summary

| Experiment | Result |
|---|---|
| Iris PC1+PC2 | ~96% variance retained |
| Digits (k=30) | Low reconstruction error |
| Digits 2D projection | Clear class separation visible |

---

## Usage

```python
from src.mlpackage.unsupervised_learning.pca import PCA
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

X_reconstructed = pca.inverse_transform(X_reduced)
```

**Key attributes after fitting:**

| Attribute | Description |
|---|---|
| `components_` | Principal axes, shape (n_components, n_features) |
| `explained_variance_` | Variance captured per component (eigenvalues) |
| `explained_variance_ratio_` | Fraction of total variance per component |
| `singular_values_` | Square root of eigenvalues |

---

## Notes

- Always **standardize features** before applying PCA. Variables with larger scales will otherwise dominate the principal components.
- Use the **scree plot** and cumulative explained variance to choose `n_components`. A common heuristic is to retain enough components to explain 95% of variance.
- `inverse_transform()` gives an approximate reconstruction — exact only when all components are retained.
- PCA captures only **linear** structure. For non-linear manifolds, consider kernel PCA or t-SNE.
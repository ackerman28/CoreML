# K-Means Clustering

A popular unsupervised learning algorithm for grouping unlabeled data.

## Features
- **Deterministic Initialization**: Supports `random_state` for reproducible results.
- **Euclidean Distance**: Uses L2 norm for cluster assignment.
- **Convergence Logic**: Includes a tolerance (`tol`) check to stop early if centroids stabilize.

## Optimization Goal
The algorithm seeks to minimize the **Within-Cluster Sum of Squares (WCSS)**:
$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$
where $\mu_i$ is the centroid of cluster $C_i$.
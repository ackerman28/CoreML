# K-Nearest Neighbors (KNN)

A memory-based (lazy learner) implementation of the KNN algorithm.

## Mathematical Formulation
The distance between two points $p$ and $q$ is calculated using the **Euclidean Distance (L2 Norm)**:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

## Features
- Custom $K$ neighbor selection.
- Majority voting using `collections.Counter`.
- Vectorized distance computation using `numpy`.
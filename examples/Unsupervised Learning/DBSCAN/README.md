# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

A density-based clustering algorithm that can discover clusters of arbitrary shapes and identify outliers.

## Parameters
- **eps ($\epsilon$)**: The radius of a neighborhood.
- **min_samples**: The minimum number of points required to form a "core point".

## Point Classifications
1. **Core Point**: Has $\ge$ `min_samples` within `eps`.
2. **Border Point**: Has fewer than `min_samples` within `eps` but is in the neighborhood of a Core Point.
3. **Noise**: Neither a core nor a border point (labeled as `-1`).
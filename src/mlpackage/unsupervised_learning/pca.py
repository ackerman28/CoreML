import numpy as np


class PCA:
    """
    Principal Component Analysis via eigendecomposition of the covariance matrix.

    PCA finds the directions of maximum variance in the data (principal components)
    and projects the data onto a lower-dimensional subspace spanned by the top
    ``n_components`` eigenvectors.

    The principal components are the eigenvectors of the covariance matrix,
    ordered by decreasing eigenvalue (variance explained).

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.

    Attributes
    ----------
    components_ : np.ndarray of shape (n_components, n_features)
        Top eigenvectors (principal axes) sorted by explained variance.
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean computed from the training data.
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance explained by each selected component (eigenvalues).
    explained_variance_ratio_ : np.ndarray of shape (n_components,)
        Fraction of total variance explained by each component.
    singular_values_ : np.ndarray of shape (n_components,)
        Square root of eigenvalues, analogous to singular values.

    Examples
    --------
    >>> from mlpackage.unsupervised_learning.pca import PCA
    >>> import numpy as np
    >>> X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=float)
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    >>> pca.transform(X).shape
    (4, 2)
    """

    def __init__(self, n_components: int):
        if n_components <= 0:
            raise ValueError(f"n_components must be a positive integer, got {n_components}")
        self.n_components = n_components
        self.components_: np.ndarray = None
        self.mean_: np.ndarray = None
        self.explained_variance_: np.ndarray = None
        self.explained_variance_ratio_: np.ndarray = None
        self.singular_values_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Compute the principal components from the training data.

        Steps:
        1. Center the data by subtracting the column-wise mean.
        2. Compute the covariance matrix.
        3. Perform eigendecomposition.
        4. Sort eigenvectors by descending eigenvalue.
        5. Store the top ``n_components`` eigenvectors.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PCA
            Fitted estimator.

        Raises
        ------
        ValueError
            If n_components exceeds the number of features.
        """
        X = np.array(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if self.n_components > X.shape[1]:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed "
                f"n_features ({X.shape[1]})"
            )

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov = np.cov(X_centered, rowvar=False)
        # Use eigh for symmetric matrices — numerically more stable than eig
        # and guarantees real eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var if total_var > 0
            else np.zeros(self.n_components)
        )
        self.singular_values_ = np.sqrt(np.maximum(self.explained_variance_, 0))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the principal components.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_projected : np.ndarray of shape (n_samples, n_components)
            Data in the reduced principal component space.

        Raises
        ------
        RuntimeError
            If called before fitting.
        """
        if self.components_ is None:
            raise RuntimeError("PCA is not fitted yet. Call fit() before transform().")
        X = np.array(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and apply dimensionality reduction in one step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_projected : np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from the principal component representation.

        The reconstruction is approximate unless all components are retained.

        Parameters
        ----------
        X_reduced : np.ndarray of shape (n_samples, n_components)
            Data in the reduced space.

        Returns
        -------
        X_reconstructed : np.ndarray of shape (n_samples, n_features)
            Approximately reconstructed data in the original feature space.

        Raises
        ------
        RuntimeError
            If called before fitting.
        """
        if self.components_ is None:
            raise RuntimeError("PCA is not fitted yet. Call fit() before inverse_transform().")
        X_reduced = np.array(X_reduced, dtype=float)
        return X_reduced @ self.components_ + self.mean_
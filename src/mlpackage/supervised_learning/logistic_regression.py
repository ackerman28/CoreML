import numpy as np


class LogisticRegression:
    """
    Logistic Regression binary classifier trained via gradient descent.

    Models the probability that a sample belongs to class 1 using the
    sigmoid function applied to a linear combination of features:

    .. math::
        P(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^\\top \\mathbf{x} + b)
        = \\frac{1}{1 + e^{-(\\mathbf{w}^\\top \\mathbf{x} + b)}}

    Parameters are learned by minimizing **binary cross-entropy loss**
    via batch gradient descent.

    Parameters
    ----------
    learning_rate : float, optional (default=0.01)
        Step size for gradient descent weight updates.
    n_iters : int, optional (default=1000)
        Number of full passes over the training data.
    threshold : float, optional (default=0.5)
        Decision threshold for converting probabilities to class labels.

    Attributes
    ----------
    weights : np.ndarray of shape (n_features,)
        Learned feature coefficients after fitting.
    bias : float
        Learned intercept term after fitting.
    loss_history : list of float
        Binary cross-entropy loss recorded after each iteration.

    Examples
    --------
    >>> from mlpackage.supervised_learning.logistic_regression import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [7, 8], [8, 9]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> model = LogisticRegression(learning_rate=0.1, n_iters=500)
    >>> model.fit(X, y)
    >>> model.predict(np.array([[10, 11]]))
    array([1])
    """

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000,
                 threshold: float = 0.5):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be a positive integer, got {n_iters}")
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")

        self.lr = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold
        self.weights: np.ndarray = None
        self.bias: float = None
        self.loss_history: list = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.

        Clips input to [-500, 500] to prevent overflow in exp.

        Parameters
        ----------
        z : np.ndarray
            Linear combination of inputs.

        Returns
        -------
        np.ndarray
            Sigmoid-transformed values in (0, 1).
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.

        .. math::
            \\mathcal{L} = -\\frac{1}{n} \\sum \\left[
                y \\log(\\hat{y}) + (1 - y) \\log(1 - \\hat{y})
            \\right]

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            True binary labels.
        y_pred : np.ndarray of shape (n_samples,)
            Predicted probabilities.

        Returns
        -------
        float
            Mean binary cross-entropy loss.
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train the logistic regression model via batch gradient descent.

        Gradients of binary cross-entropy with respect to weights and bias:

        .. math::
            \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{w}} =
            \\frac{1}{n} X^\\top (\\hat{y} - y), \\quad
            \\frac{\\partial \\mathcal{L}}{\\partial b} =
            \\frac{1}{n} \\sum (\\hat{y} - y)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Binary class labels (0 or 1).

        Returns
        -------
        self : LogisticRegression
            Fitted estimator (enables method chaining).

        Raises
        ------
        ValueError
            If X and y have inconsistent numbers of samples.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.loss_history.append(self._binary_cross_entropy(y, y_pred))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : np.ndarray of shape (n_samples,)
            Estimated probability of belonging to class 1.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted.
        """
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict_proba().")
        X = np.array(X, dtype=float)
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted binary class labels (0 or 1).

        Raises
        ------
        RuntimeError
            If called before the model has been fitted.
        """
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            True binary labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples (0.0 to 1.0).
        """
        return float(np.mean(self.predict(X) == np.array(y, dtype=int)))
import numpy as np


class LinearRegression:
    """
    Linear Regression implemented via the Normal Equation and Gradient Descent.

    Two solvers are available:

    - ``'normal'`` — Closed-form solution using the Normal Equation:

      .. math::
          \\boldsymbol{\\theta} = (X^\\top X)^{-1} X^\\top \\mathbf{y}

      Exact and efficient for small-to-medium datasets. May be numerically
      unstable when features are highly collinear.

    - ``'gradient_descent'`` — Iterative optimization via batch gradient descent.
      Scales better to large datasets and avoids matrix inversion.

    Parameters
    ----------
    method : str, optional (default='normal')
        Solver to use. One of ``{'normal', 'gradient_descent'}``.
    learning_rate : float, optional (default=0.01)
        Step size for gradient descent updates. Ignored when method='normal'.
    n_iters : int, optional (default=1000)
        Number of gradient descent iterations. Ignored when method='normal'.

    Attributes
    ----------
    weights : np.ndarray of shape (n_features,)
        Learned feature coefficients after fitting.
    bias : float
        Learned intercept term after fitting.
    loss_history : list of float
        MSE recorded after each gradient descent iteration.
        Empty when method='normal'.

    Examples
    --------
    >>> from mlpackage.supervised_learning.linear_regression import LinearRegression
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([3.0, 5.0, 7.0, 9.0])
    >>> model = LinearRegression(method='normal')
    >>> model.fit(X, y)
    >>> model.predict(np.array([[5]]))
    array([11.])
    """

    def __init__(self, method: str = "normal", learning_rate: float = 0.01,
                 n_iters: int = 1000):
        if method not in ("normal", "gradient_descent"):
            raise ValueError(
                f"method must be 'normal' or 'gradient_descent', got '{method}'"
            )
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be a positive integer, got {n_iters}")

        self.method = method
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights: np.ndarray = None
        self.bias: float = None
        self.loss_history: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the linear model to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self : LinearRegression
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

        if self.method == "normal":
            self._fit_normal(X, y)
        else:
            self._fit_gradient_descent(X, y)

        return self

    def _fit_normal(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Solve via the Normal Equation.

        Augments X with a bias column of ones, then computes:
        theta = (X^T X)^{-1} X^T y using numpy's pseudoinverse (pinv)
        for numerical stability.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        X_b = np.c_[np.ones(X.shape[0]), X]
        # Use pinv instead of inv for numerical stability with collinear features
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = float(theta[0])
        self.weights = theta[1:]

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Solve via batch gradient descent.

        Gradients with respect to weights and bias:

        dL/dw = -(2/n) * X^T (y - y_hat)
        dL/db = -(2/n) * sum(y - y_hat)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            residuals = y - y_pred

            dw = -(2 / n_samples) * X.T @ residuals
            db = -(2 / n_samples) * np.sum(residuals)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            mse = np.mean(residuals ** 2)
            self.loss_history.append(mse)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for input samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted.
        """
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")

        X = np.array(X, dtype=float)
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the coefficient of determination R² on the given data.

        R² = 1 - SS_res / SS_tot, where SS_res is the residual sum of squares
        and SS_tot is the total sum of squares. A score of 1.0 is perfect;
        0.0 means the model performs no better than predicting the mean.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R² score.
        """
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
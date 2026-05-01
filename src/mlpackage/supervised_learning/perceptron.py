import numpy as np


class Perceptron:
    """
    Single-layer Perceptron classifier trained via the Perceptron Learning Rule.

    The Perceptron is a binary linear classifier. It learns a decision boundary
    by iterating over training samples and updating weights whenever a sample is
    misclassified. Convergence is guaranteed only when data is linearly separable.

    Parameters
    ----------
    learning_rate : float, optional (default=0.01)
        Step size applied during each weight update. Larger values converge
        faster but may overshoot; smaller values are more stable.
    n_iters : int, optional (default=1000)
        Maximum number of full passes (epochs) over the training dataset.

    Attributes
    ----------
    weights : np.ndarray of shape (n_features,)
        Learned feature weights after training.
    bias : float
        Learned bias term after training.
    training_errors : list of int
        Number of misclassifications per epoch. Useful for diagnosing convergence.

    Examples
    --------
    >>> from mlpackage.supervised_learning.perceptron import Perceptron
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = Perceptron(learning_rate=0.1, n_iters=100)
    >>> clf.fit(X, y)
    >>> clf.predict(np.array([[1.5, 2.5], [3.5, 4.5]]))
    array([0, 1])
    """

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be a positive integer, got {n_iters}")

        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights: np.ndarray = None
        self.bias: float = None
        self.training_errors: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train the Perceptron on labeled binary data.

        For each epoch, the algorithm iterates over every sample. If the
        predicted label differs from the true label, weights and bias are
        updated by a scaled error signal. No update occurs on correct predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Binary class labels. Any positive value is mapped to 1; others to 0.

        Returns
        -------
        self : Perceptron
            Fitted estimator (enables method chaining).

        Raises
        ------
        ValueError
            If X and y have inconsistent numbers of samples.
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

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
        self.training_errors = []

        # Map labels to binary {0, 1}
        y_binary = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                y_predicted = self._activation(np.dot(x_i, self.weights) + self.bias)
                update = self.lr * (y_binary[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0)
            self.training_errors.append(errors)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input samples.

        Computes the linear combination of inputs and learned weights,
        then applies the Heaviside step function as the activation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to classify.

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

        X = np.array(X, dtype=float)
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

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
        y_binary = np.where(np.array(y) > 0, 1, 0)
        return np.mean(self.predict(X) == y_binary)

    def _activation(self, linear_output: float) -> int:
        """
        Heaviside step function used as the Perceptron's activation.

        Parameters
        ----------
        linear_output : float
            Raw linear combination (w · x + b).

        Returns
        -------
        int
            1 if linear_output >= 0, else 0.
        """
        return 1 if linear_output >= 0 else 0
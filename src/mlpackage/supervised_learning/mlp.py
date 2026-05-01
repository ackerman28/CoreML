import numpy as np


class MultilayerPerceptron:
    """
    Fully-connected feedforward neural network trained via backpropagation.

    Supports a single hidden layer with sigmoid activations on both the hidden
    and output layers. Suitable for binary classification tasks.

    The forward pass computes:

    .. math::
        \\mathbf{a}^{(1)} = \\sigma(X W^{(1)} + \\mathbf{b}^{(1)}), \\quad
        \\mathbf{a}^{(2)} = \\sigma(\\mathbf{a}^{(1)} W^{(2)} + \\mathbf{b}^{(2)})

    Weights are updated via backpropagation using mean squared error loss.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of neurons in the hidden layer.
    output_size : int
        Number of output neurons (1 for binary classification).
    learning_rate : float, optional (default=0.1)
        Step size for gradient descent weight updates.
    n_iters : int, optional (default=1000)
        Number of training epochs.
    random_state : int or None, optional (default=42)
        Seed for reproducible weight initialization.

    Attributes
    ----------
    W1 : np.ndarray of shape (input_size, hidden_size)
        Weight matrix for the input-to-hidden layer.
    b1 : np.ndarray of shape (1, hidden_size)
        Bias vector for the hidden layer.
    W2 : np.ndarray of shape (hidden_size, output_size)
        Weight matrix for the hidden-to-output layer.
    b2 : np.ndarray of shape (1, output_size)
        Bias vector for the output layer.
    loss_history : list of float
        MSE loss recorded after each epoch.

    Examples
    --------
    >>> from mlpackage.supervised_learning.multilayer_perceptron import MultilayerPerceptron
    >>> import numpy as np
    >>> X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    >>> y = np.array([[0],[1],[1],[0]])
    >>> model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
    ...                              learning_rate=0.5, n_iters=10000)
    >>> model.fit(X, y)
    >>> model.predict(X)
    array([[0],[1],[1],[0]])
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.1, n_iters: int = 1000,
                 random_state: int = 42):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be a positive integer, got {n_iters}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state
        self.loss_history: list = []

        rng = np.random.default_rng(random_state)
        # Xavier-style initialization: scale by 1/sqrt(fan_in)
        self.W1 = rng.normal(0, 1 / np.sqrt(input_size),  (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid activation function.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values.

        Returns
        -------
        np.ndarray
            Sigmoid-activated values in (0, 1).
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid given its output (post-activation).

        .. math::
            \\sigma'(z) = \\sigma(z)(1 - \\sigma(z)) = a(1 - a)

        Parameters
        ----------
        a : np.ndarray
            Post-activation values (output of sigmoid).

        Returns
        -------
        np.ndarray
            Element-wise derivative values.
        """
        return a * (1 - a)

    def _forward(self, X: np.ndarray) -> tuple:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, input_size)
            Input feature matrix.

        Returns
        -------
        a1 : np.ndarray of shape (n_samples, hidden_size)
            Hidden layer activations.
        a2 : np.ndarray of shape (n_samples, output_size)
            Output layer activations (predicted probabilities).
        """
        z1 = X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)
        return a1, a2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultilayerPerceptron":
        """
        Train the MLP via backpropagation.

        For each epoch, performs a full forward pass then backpropagates
        the error signal to update all weights and biases.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, input_size)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,) or (n_samples, output_size)
            Binary target labels.

        Returns
        -------
        self : MultilayerPerceptron
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
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        self.loss_history = []

        for _ in range(self.n_iters):
            # Forward pass
            a1, a2 = self._forward(X)

            # Backward pass
            error_output = y - a2
            d_output = error_output * self._sigmoid_derivative(a2)

            error_hidden = d_output @ self.W2.T
            d_hidden = error_hidden * self._sigmoid_derivative(a1)

            # Weight updates
            self.W2 += self.lr * a1.T @ d_output
            self.b2 += self.lr * np.sum(d_output, axis=0, keepdims=True)
            self.W1 += self.lr * X.T @ d_hidden
            self.b1 += self.lr * np.sum(d_hidden, axis=0, keepdims=True)

            mse = float(np.mean((y - a2) ** 2))
            self.loss_history.append(mse)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return output layer activations as predicted probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, input_size)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples, output_size)
            Predicted probabilities in (0, 1).
        """
        X = np.array(X, dtype=float)
        _, a2 = self._forward(X)
        return a2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels by thresholding output probabilities at 0.5.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, input_size)
            Feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples, output_size)
            Predicted binary labels (0 or 1).
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, input_size)
            Feature matrix.
        y : np.ndarray of shape (n_samples,) or (n_samples, output_size)
            True binary labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples (0.0 to 1.0).
        """
        y = np.array(y, dtype=int)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return float(np.mean(self.predict(X) == y))
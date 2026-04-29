import numpy as np

class Perceptron:
    """
    A from-scratch implementation of the Perceptron algorithm.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        # learning_rate: how much we change the weights each step
        # n_iters: how many times we look at the whole dataset
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model using the Perceptron Update Rule.
        """
        n_samples, n_features = X.shape

        # Start with weights as zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # The Perceptron expects labels to be 0 or 1
        # This line ensures the labels are in that format
        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate: (weights * input) + bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Activation: If result >= 0, predict 1, else 0
                y_predicted = 1 if linear_output >= 0 else 0

                # The "Learning" step:
                # If we were wrong, update weights. If right, update is 0.
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predict labels for new data.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
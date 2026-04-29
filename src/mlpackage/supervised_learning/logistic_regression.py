import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier using Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # 1. Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # 2. Apply sigmoid
            y_predicted = self._sigmoid(linear_model)

            # 3. Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 4. Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        y_predicted_cls = [1 if i > 0.5 else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)
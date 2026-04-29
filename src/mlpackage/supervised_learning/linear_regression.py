import numpy as np

class LinearRegression:
    """
    Linear Regression using the Normal Equation.
    """
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model using the Normal Equation: 
        theta = (X_transpose * X)^-1 * X_transpose * y
        """
        # Add a column of ones to X to account for the bias (intercept)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate the weights using the Normal Equation
        # theta_best = inv(X_b.T . X_b) . X_b.T . y
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.bias = theta_best[0]
        self.weights = theta_best[1:]

    def predict(self, X):
        """Make predictions using the linear model."""
        return np.dot(X, self.weights) + self.bias
    
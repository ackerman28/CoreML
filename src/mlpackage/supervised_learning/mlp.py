import numpy as np

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        
        # Seed for reproducibility
        np.random.seed(42) 
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        # x is already the sigmoid output
        return x * (1 - x)

    def fit(self, X, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        for i in range(self.n_iters):
            # Forward Pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._sigmoid(z1)
            
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._sigmoid(z2)
            
            # Backward Pass
            error_output = y - a2
            d_output = error_output * self._sigmoid_derivative(a2)
            
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self._sigmoid_derivative(a1)
            
            # Weight Updates
            self.W2 += a1.T.dot(d_output) * self.lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
            self.W1 += X.T.dot(d_hidden) * self.lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

    def predict(self, X):
        """The missing method: passes input through the network to get a class prediction."""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)
        # Threshold at 0.5 to get 0 or 1
        return (a2 > 0.5).astype(int)
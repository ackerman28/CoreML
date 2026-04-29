import numpy as np
from mlpackage import MultilayerPerceptron

def test_mlp_xor():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Increase iterations or LR if it still struggles
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5, n_iters=20000)
    model.fit(X, y)
    
    predictions = model.predict(X)
    np.testing.assert_array_equal(predictions, y)
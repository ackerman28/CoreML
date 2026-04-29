import numpy as np
from mlpackage import Perceptron

def test_perceptron_and_gate():
    """Test if the Perceptron can learn the AND logic gate."""
    # Inputs: [0,0], [0,1], [1,0], [1,1]
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Targets for AND: Only [1,1] results in 1
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    # This 'assert' is what pytest looks for
    assert np.array_equal(predictions, y), f"Failed! Got {predictions}"

if __name__ == "__main__":
    test_perceptron_and_gate()
    print("Test Passed Locally!")
import numpy as np
from mlpackage import LogisticRegression

def test_logistic_regression():
    # Simple linearly separable data
    X = np.array([[1, 2], [2, 3], [3, 4], [7, 8], [8, 9], [9, 10]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    
    # Predict a clearly "class 1" point
    prediction = model.predict(np.array([[10, 11]]))
    assert prediction[0] == 1
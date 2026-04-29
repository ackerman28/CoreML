import numpy as np
from mlpackage import LinearRegression, mean_squared_error

def test_linear_regression_simple():
    # Create data: y = 2x + 1
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict for x=5 (should be 11)
    prediction = model.predict(np.array([[5]]))
    
    assert np.isclose(prediction[0], 11.0), f"Expected 11.0, got {prediction[0]}"
    assert np.isclose(model.bias, 1.0), f"Expected bias 1.0, got {model.bias}"

if __name__ == "__main__":
    test_linear_regression_simple()
    print("Linear Regression Test Passed!")
    
import numpy as np
import pytest
from mlpackage import LinearRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_line():
    """Noise-free y = 2x + 1."""
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)
    return X, y


@pytest.fixture
def multivariate():
    """y = 3x1 + 2x2 + 5, noise-free."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    y = X[:, 0] * 3 + X[:, 1] * 2 + 5
    return X, y


@pytest.fixture
def noisy_data():
    """Noisy linear data for regression diagnostics."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(100, 1))
    y = 4 * X.squeeze() + 2 + rng.normal(0, 1, 100)
    return X, y


# ---------------------------------------------------------------------------
# Normal Equation — correctness
# ---------------------------------------------------------------------------

def test_normal_perfect_fit(perfect_line):
    """Normal equation should recover exact coefficients on noise-free data."""
    X, y = perfect_line
    model = LinearRegression(method="normal")
    model.fit(X, y)
    assert np.isclose(model.bias, 1.0, atol=1e-6)
    assert np.isclose(model.weights[0], 2.0, atol=1e-6)


def test_normal_predict_value(perfect_line):
    """Predicted value for x=6 on y=2x+1 should be 13."""
    X, y = perfect_line
    model = LinearRegression(method="normal")
    model.fit(X, y)
    pred = model.predict(np.array([[6.0]]))
    assert np.isclose(pred[0], 13.0, atol=1e-6)


def test_normal_multivariate(multivariate):
    """Normal equation should produce accurate predictions on multivariate input."""
    X, y = multivariate
    model = LinearRegression(method="normal")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, y, atol=1e-4)


def test_normal_r2_perfect(perfect_line):
    """R² should be 1.0 on noise-free linear data."""
    X, y = perfect_line
    model = LinearRegression(method="normal")
    model.fit(X, y)
    assert np.isclose(model.score(X, y), 1.0, atol=1e-6)


def test_normal_r2_noisy(noisy_data):
    """R² should be high (>0.95) on low-noise data."""
    X, y = noisy_data
    model = LinearRegression(method="normal")
    model.fit(X, y)
    assert model.score(X, y) > 0.95


# ---------------------------------------------------------------------------
# Gradient Descent — correctness
# ---------------------------------------------------------------------------

def test_gd_converges_on_perfect_line(perfect_line):
    """Gradient descent should closely recover weights on noise-free data."""
    X, y = perfect_line
    model = LinearRegression(method="gradient_descent", learning_rate=0.01, n_iters=5000)
    model.fit(X, y)
    assert np.isclose(model.bias, 1.0, atol=0.05)
    assert np.isclose(model.weights[0], 2.0, atol=0.05)


def test_gd_loss_decreases(noisy_data):
    """Loss should be strictly decreasing over gradient descent iterations."""
    X, y = noisy_data
    model = LinearRegression(method="gradient_descent", learning_rate=0.01, n_iters=200)
    model.fit(X, y)
    losses = model.loss_history
    assert losses[0] > losses[-1], "Loss should decrease over training"


def test_gd_loss_history_length(noisy_data):
    """loss_history should have exactly n_iters entries."""
    X, y = noisy_data
    n_iters = 150
    model = LinearRegression(method="gradient_descent", n_iters=n_iters)
    model.fit(X, y)
    assert len(model.loss_history) == n_iters


def test_gd_r2_noisy(noisy_data):
    """Gradient descent R² should be high on low-noise linear data."""
    X, y = noisy_data
    model = LinearRegression(method="gradient_descent", learning_rate=0.01, n_iters=2000)
    model.fit(X, y)
    assert model.score(X, y) > 0.93


def test_normal_loss_history_empty(perfect_line):
    """Normal equation solver should leave loss_history empty."""
    X, y = perfect_line
    model = LinearRegression(method="normal")
    model.fit(X, y)
    assert model.loss_history == []


# ---------------------------------------------------------------------------
# predict() & score()
# ---------------------------------------------------------------------------

def test_predict_output_shape(noisy_data):
    """predict() output shape should match number of input samples."""
    X, y = noisy_data
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_score_range(noisy_data):
    """R² score should be <= 1.0."""
    X, y = noisy_data
    model = LinearRegression()
    model.fit(X, y)
    assert model.score(X, y) <= 1.0


def test_score_matches_manual_r2(perfect_line):
    """score() should match manually computed R²."""
    X, y = perfect_line
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    manual_r2 = 1 - ss_res / ss_tot
    assert np.isclose(model.score(X, y), manual_r2, atol=1e-8)


# ---------------------------------------------------------------------------
# Weights & Bias
# ---------------------------------------------------------------------------

def test_weights_shape(multivariate):
    """Weights shape should match number of features."""
    X, y = multivariate
    model = LinearRegression()
    model.fit(X, y)
    assert model.weights.shape == (X.shape[1],)


def test_bias_is_float(perfect_line):
    """Bias should be a Python float after fitting."""
    X, y = perfect_line
    model = LinearRegression()
    model.fit(X, y)
    assert isinstance(model.bias, float)


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """Calling predict() before fit() should raise RuntimeError."""
    model = LinearRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_invalid_method():
    """Unknown method string should raise ValueError."""
    with pytest.raises(ValueError):
        LinearRegression(method="svd")


def test_invalid_learning_rate():
    """Non-positive learning rate should raise ValueError."""
    with pytest.raises(ValueError):
        LinearRegression(learning_rate=0)
    with pytest.raises(ValueError):
        LinearRegression(learning_rate=-0.1)


def test_invalid_n_iters():
    """Non-positive n_iters should raise ValueError."""
    with pytest.raises(ValueError):
        LinearRegression(n_iters=0)
    with pytest.raises(ValueError):
        LinearRegression(n_iters=-5)


def test_mismatched_X_y_raises():
    """Mismatched X and y sample counts should raise ValueError."""
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])
    with pytest.raises(ValueError):
        LinearRegression().fit(X, y)


def test_single_sample():
    """Single-sample input should not crash the model."""
    X = np.array([[3.0]])
    y = np.array([7.0])
    model = LinearRegression(method="normal")
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (1,)


def test_method_chaining(perfect_line):
    """fit() should return self to support method chaining."""
    X, y = perfect_line
    model = LinearRegression()
    result = model.fit(X, y)
    assert result is model
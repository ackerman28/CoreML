import numpy as np
import pytest
from mlpackage import Perceptron


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def and_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    return X, y


@pytest.fixture
def or_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1])
    return X, y


@pytest.fixture
def linearly_separable():
    """Two well-separated 2D Gaussian clusters."""
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=-3, scale=0.5, size=(30, 2))
    X1 = rng.normal(loc=+3, scale=0.5, size=(30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 30 + [1] * 30)
    return X, y


# ---------------------------------------------------------------------------
# Logic Gate Tests
# ---------------------------------------------------------------------------

def test_and_gate(and_gate):
    """Perceptron should learn the AND logic gate perfectly."""
    X, y = and_gate
    model = Perceptron(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_or_gate(or_gate):
    """Perceptron should learn the OR logic gate perfectly."""
    X, y = or_gate
    model = Perceptron(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_xor_gate_does_not_converge():
    """XOR is not linearly separable; Perceptron should not achieve 100% accuracy."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = Perceptron(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    accuracy = model.score(X, y)
    assert accuracy < 1.0, "Perceptron should not perfectly solve XOR"


# ---------------------------------------------------------------------------
# Convergence & Training Diagnostics
# ---------------------------------------------------------------------------

def test_convergence_on_separable_data(linearly_separable):
    """Perceptron should reach zero training errors on linearly separable data."""
    X, y = linearly_separable
    model = Perceptron(learning_rate=0.1, n_iters=200)
    model.fit(X, y)
    assert model.training_errors[-1] == 0, (
        f"Expected 0 errors in final epoch, got {model.training_errors[-1]}"
    )


def test_training_errors_recorded(and_gate):
    """training_errors should have exactly n_iters entries after fitting."""
    X, y = and_gate
    n_iters = 50
    model = Perceptron(n_iters=n_iters)
    model.fit(X, y)
    assert len(model.training_errors) == n_iters


def test_high_accuracy_on_separable_data(linearly_separable):
    """Perceptron should achieve perfect accuracy on linearly separable data."""
    X, y = linearly_separable
    model = Perceptron(learning_rate=0.1, n_iters=200)
    model.fit(X, y)
    assert model.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# Weights & Bias
# ---------------------------------------------------------------------------

def test_weights_shape(and_gate):
    """Weights should match the number of input features."""
    X, y = and_gate
    model = Perceptron()
    model.fit(X, y)
    assert model.weights.shape == (X.shape[1],)


def test_bias_is_scalar(and_gate):
    """Bias should be a plain Python float after fitting."""
    X, y = and_gate
    model = Perceptron()
    model.fit(X, y)
    assert isinstance(model.bias, float)


def test_weights_change_after_fit(and_gate):
    """Weights should be non-zero after training on non-trivial data."""
    X, y = and_gate
    model = Perceptron(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    assert not np.all(model.weights == 0), "Weights should update during training"


# ---------------------------------------------------------------------------
# predict() & score()
# ---------------------------------------------------------------------------

def test_predict_output_shape(linearly_separable):
    """predict() should return an array with the same length as input."""
    X, y = linearly_separable
    model = Perceptron()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_predict_binary_outputs(linearly_separable):
    """All predicted values should be exactly 0 or 1."""
    X, y = linearly_separable
    model = Perceptron()
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_score_range(linearly_separable):
    """score() should always return a value in [0, 1]."""
    X, y = linearly_separable
    model = Perceptron()
    model.fit(X, y)
    s = model.score(X, y)
    assert 0.0 <= s <= 1.0


def test_score_matches_manual_accuracy(and_gate):
    """score() should match manually computed accuracy."""
    X, y = and_gate
    model = Perceptron(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    preds = model.predict(X)
    manual_acc = np.mean(preds == y)
    assert model.score(X, y) == pytest.approx(manual_acc)


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """Calling predict() before fit() should raise RuntimeError."""
    model = Perceptron()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, 2]]))


def test_invalid_learning_rate():
    """Non-positive learning rate should raise ValueError."""
    with pytest.raises(ValueError):
        Perceptron(learning_rate=0)
    with pytest.raises(ValueError):
        Perceptron(learning_rate=-0.5)


def test_invalid_n_iters():
    """Non-positive n_iters should raise ValueError."""
    with pytest.raises(ValueError):
        Perceptron(n_iters=0)
    with pytest.raises(ValueError):
        Perceptron(n_iters=-10)


def test_mismatched_X_y_raises():
    """Mismatched X and y sample counts should raise ValueError."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    model = Perceptron()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_single_sample():
    """Model should handle a single training sample without error."""
    X = np.array([[1.0, 2.0]])
    y = np.array([1])
    model = Perceptron(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (1,)


def test_all_same_label():
    """All-same-class data should not crash the model."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    y = np.array([1, 1, 1])
    model = Perceptron(learning_rate=0.1, n_iters=50)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (3,)


def test_positive_labels_mapped_to_one():
    """Labels like -1/+1 should be handled; +1 maps to 1, -1 maps to 0."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_signed = np.array([-1, -1, -1, 1])
    y_binary = np.array([0, 0, 0, 1])
    model = Perceptron(learning_rate=0.1, n_iters=100)
    model.fit(X, y_signed)
    assert np.array_equal(model.predict(X), y_binary)
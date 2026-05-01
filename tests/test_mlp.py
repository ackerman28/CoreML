import numpy as np
import pytest
from mlpackage import MultilayerPerceptron


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]])
    return X, y


@pytest.fixture
def and_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [0], [0], [1]])
    return X, y


@pytest.fixture
def gaussian_clusters():
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=-2, scale=0.5, size=(40, 2))
    X1 = rng.normal(loc=+2, scale=0.5, size=(40, 2))
    X = np.vstack([X0, X1])
    y = np.array([[0]] * 40 + [[1]] * 40)
    return X, y


# ---------------------------------------------------------------------------
# Core Functionality
# ---------------------------------------------------------------------------

def test_xor_solved(xor_data):
    """MLP should solve XOR — a non-linearly separable problem."""
    X, y = xor_data
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  learning_rate=0.5, n_iters=20000)
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)


def test_and_gate(and_data):
    """MLP should learn AND gate."""
    X, y = and_data
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  learning_rate=0.5, n_iters=5000)
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)


def test_high_accuracy_gaussian(gaussian_clusters):
    """MLP should achieve >95% accuracy on linearly separable clusters."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=8, output_size=1,
                                  learning_rate=0.1, n_iters=3000)
    model.fit(X, y)
    assert model.score(X, y) > 0.95


# ---------------------------------------------------------------------------
# Loss & Convergence
# ---------------------------------------------------------------------------

def test_loss_decreases(gaussian_clusters):
    """MSE loss should decrease over training epochs."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=8, output_size=1,
                                  learning_rate=0.1, n_iters=500)
    model.fit(X, y)
    assert model.loss_history[0] > model.loss_history[-1]


def test_loss_history_length(gaussian_clusters):
    """loss_history should have exactly n_iters entries."""
    X, y = gaussian_clusters
    n_iters = 120
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=n_iters)
    model.fit(X, y)
    assert len(model.loss_history) == n_iters


def test_loss_is_positive(gaussian_clusters):
    """All loss values should be non-negative."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=100)
    model.fit(X, y)
    assert all(l >= 0 for l in model.loss_history)


# ---------------------------------------------------------------------------
# predict() & predict_proba() & score()
# ---------------------------------------------------------------------------

def test_predict_shape(gaussian_clusters):
    """predict() should return shape (n_samples, output_size)."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=100)
    model.fit(X, y)
    assert model.predict(X).shape == (X.shape[0], 1)


def test_predict_binary_values(gaussian_clusters):
    """All predicted values should be exactly 0 or 1."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_proba_range(gaussian_clusters):
    """predict_proba() values should all be in (0, 1)."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=200)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba > 0) and np.all(proba < 1)


def test_predict_proba_shape(gaussian_clusters):
    """predict_proba() should return shape (n_samples, output_size)."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=100)
    model.fit(X, y)
    assert model.predict_proba(X).shape == (X.shape[0], 1)


def test_score_range(gaussian_clusters):
    """score() should return a value in [0, 1]."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=200)
    model.fit(X, y)
    assert 0.0 <= model.score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# Weight Initialization & Architecture
# ---------------------------------------------------------------------------

def test_weight_shapes():
    """W1, W2, b1, b2 should have correct shapes after initialization."""
    model = MultilayerPerceptron(input_size=3, hidden_size=5, output_size=1)
    assert model.W1.shape == (3, 5)
    assert model.b1.shape == (1, 5)
    assert model.W2.shape == (5, 1)
    assert model.b2.shape == (1, 1)


def test_weights_change_after_fit(gaussian_clusters):
    """Weights should change from their initial values after training."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=100)
    W1_init = model.W1.copy()
    model.fit(X, y)
    assert not np.allclose(model.W1, W1_init)


def test_reproducibility():
    """Same random_state should produce identical weight initialization."""
    m1 = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1, random_state=7)
    m2 = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1, random_state=7)
    assert np.allclose(m1.W1, m2.W1)
    assert np.allclose(m1.W2, m2.W2)


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_invalid_learning_rate():
    """Non-positive learning rate should raise ValueError."""
    with pytest.raises(ValueError):
        MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1, learning_rate=0)
    with pytest.raises(ValueError):
        MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1, learning_rate=-0.1)


def test_invalid_n_iters():
    """Non-positive n_iters should raise ValueError."""
    with pytest.raises(ValueError):
        MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1, n_iters=0)


def test_invalid_hidden_size():
    """Non-positive hidden_size should raise ValueError."""
    with pytest.raises(ValueError):
        MultilayerPerceptron(input_size=2, hidden_size=0, output_size=1)


def test_mismatched_X_y_raises(xor_data):
    """Mismatched X and y sample counts should raise ValueError."""
    X, y = xor_data
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1)
    with pytest.raises(ValueError):
        model.fit(X, y[:2])


def test_1d_y_accepted(gaussian_clusters):
    """1D y array should be reshaped internally without error."""
    X, y = gaussian_clusters
    y_flat = y.squeeze()
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=100)
    model.fit(X, y_flat)
    assert model.predict(X).shape == (X.shape[0], 1)


def test_method_chaining(gaussian_clusters):
    """fit() should return self to support method chaining."""
    X, y = gaussian_clusters
    model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1,
                                  n_iters=50)
    assert model.fit(X, y) is model
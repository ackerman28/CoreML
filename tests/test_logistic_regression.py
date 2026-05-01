import numpy as np
import pytest
from mlpackage import LogisticRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separable_2d():
    """Two clearly separated 2D clusters."""
    X = np.array([[1, 2], [2, 3], [3, 4], [7, 8], [8, 9], [9, 10]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def gaussian_clusters():
    """Two well-separated Gaussian clusters."""
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=-3, scale=0.5, size=(50, 2))
    X1 = rng.normal(loc=+3, scale=0.5, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def breast_cancer_data():
    """Breast cancer dataset split and standardized."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_predicts_class_one(separable_2d):
    """Should predict class 1 for a clearly class-1 point."""
    X, y = separable_2d
    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    assert model.predict(np.array([[10, 11]]))[0] == 1


def test_predicts_class_zero(separable_2d):
    """Should predict class 0 for a clearly class-0 point."""
    X, y = separable_2d
    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    assert model.predict(np.array([[-2, -1]]))[0] == 0


def test_perfect_accuracy_separable(separable_2d):
    """Should achieve 100% accuracy on clearly separable data."""
    X, y = separable_2d
    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    assert model.score(X, y) == 1.0


def test_high_accuracy_gaussian(gaussian_clusters):
    """Should achieve >99% accuracy on well-separated Gaussian clusters."""
    X, y = gaussian_clusters
    model = LogisticRegression(learning_rate=0.1, n_iters=500)
    model.fit(X, y)
    assert model.score(X, y) > 0.99


def test_real_dataset_accuracy(breast_cancer_data):
    """Should achieve >90% accuracy on breast cancer test set."""
    X_tr, X_te, y_tr, y_te = breast_cancer_data
    model = LogisticRegression(learning_rate=0.1, n_iters=500)
    model.fit(X_tr, y_tr)
    assert model.score(X_te, y_te) > 0.90


# ---------------------------------------------------------------------------
# Loss & Convergence
# ---------------------------------------------------------------------------

def test_loss_decreases(gaussian_clusters):
    """Binary cross-entropy loss should decrease over training."""
    X, y = gaussian_clusters
    model = LogisticRegression(learning_rate=0.1, n_iters=300)
    model.fit(X, y)
    assert model.loss_history[0] > model.loss_history[-1]


def test_loss_history_length(gaussian_clusters):
    """loss_history should have exactly n_iters entries."""
    X, y = gaussian_clusters
    n_iters = 75
    model = LogisticRegression(n_iters=n_iters)
    model.fit(X, y)
    assert len(model.loss_history) == n_iters


def test_loss_is_positive(gaussian_clusters):
    """All loss values should be positive."""
    X, y = gaussian_clusters
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    assert all(l > 0 for l in model.loss_history)


# ---------------------------------------------------------------------------
# predict_proba()
# ---------------------------------------------------------------------------

def test_proba_range(gaussian_clusters):
    """Predicted probabilities should be in (0, 1)."""
    X, y = gaussian_clusters
    model = LogisticRegression(learning_rate=0.1, n_iters=300)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_proba_shape(gaussian_clusters):
    """predict_proba() should return shape (n_samples,)."""
    X, y = gaussian_clusters
    model = LogisticRegression(learning_rate=0.1, n_iters=200)
    model.fit(X, y)
    assert model.predict_proba(X).shape == (X.shape[0],)


def test_proba_above_threshold_for_class1(separable_2d):
    """Class-1 samples should have predicted probability > 0.5."""
    X, y = separable_2d
    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    proba = model.predict_proba(X[3:])
    assert np.all(proba > 0.5)


# ---------------------------------------------------------------------------
# predict() & score()
# ---------------------------------------------------------------------------

def test_predict_output_shape(gaussian_clusters):
    """predict() should return shape (n_samples,)."""
    X, y = gaussian_clusters
    model = LogisticRegression(learning_rate=0.1, n_iters=200)
    model.fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)


def test_predict_binary_values(gaussian_clusters):
    """All predicted values should be exactly 0 or 1."""
    X, y = gaussian_clusters
    model = LogisticRegression(learning_rate=0.1, n_iters=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_score_range(gaussian_clusters):
    """score() should return a value in [0, 1]."""
    X, y = gaussian_clusters
    model = LogisticRegression(learning_rate=0.1, n_iters=200)
    model.fit(X, y)
    assert 0.0 <= model.score(X, y) <= 1.0


def test_custom_threshold():
    """Higher threshold should classify fewer samples as class 1."""
    X = np.array([[1, 2], [2, 3], [3, 4], [7, 8], [8, 9], [9, 10]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    model_default = LogisticRegression(learning_rate=0.1, n_iters=1000, threshold=0.5)
    model_strict  = LogisticRegression(learning_rate=0.1, n_iters=1000, threshold=0.9)
    model_default.fit(X, y)
    model_strict.fit(X, y)
    assert model_strict.predict(X).sum() <= model_default.predict(X).sum()


# ---------------------------------------------------------------------------
# Weights & Bias
# ---------------------------------------------------------------------------

def test_weights_shape(separable_2d):
    """Weights shape should match number of input features."""
    X, y = separable_2d
    model = LogisticRegression()
    model.fit(X, y)
    assert model.weights.shape == (X.shape[1],)


def test_bias_is_float(separable_2d):
    """Bias should be a Python float after fitting."""
    X, y = separable_2d
    model = LogisticRegression()
    model.fit(X, y)
    assert isinstance(model.bias, float)


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict() before fit() should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        LogisticRegression().predict(np.array([[1, 2]]))


def test_predict_proba_before_fit_raises():
    """predict_proba() before fit() should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_proba(np.array([[1, 2]]))


def test_invalid_learning_rate():
    """Non-positive learning rate should raise ValueError."""
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=0)
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=-0.1)


def test_invalid_n_iters():
    """Non-positive n_iters should raise ValueError."""
    with pytest.raises(ValueError):
        LogisticRegression(n_iters=0)
    with pytest.raises(ValueError):
        LogisticRegression(n_iters=-5)


def test_invalid_threshold():
    """Threshold outside (0, 1) should raise ValueError."""
    with pytest.raises(ValueError):
        LogisticRegression(threshold=0.0)
    with pytest.raises(ValueError):
        LogisticRegression(threshold=1.0)
    with pytest.raises(ValueError):
        LogisticRegression(threshold=1.5)


def test_mismatched_X_y_raises():
    """Mismatched X and y sample counts should raise ValueError."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)


def test_method_chaining(separable_2d):
    """fit() should return self to support method chaining."""
    X, y = separable_2d
    model = LogisticRegression()
    assert model.fit(X, y) is model


def test_single_sample():
    """Single-sample input should not crash the model."""
    X = np.array([[3.0, 4.0]])
    y = np.array([1])
    model = LogisticRegression(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    assert model.predict(X).shape == (1,)
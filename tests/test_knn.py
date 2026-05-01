import numpy as np
import pytest
from mlpackage.supervised_learning.knn import KNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_cluster():
    """Two clearly separated clusters."""
    X_train = np.array([[0,0],[1,1],[0.5,0.5],[5,5],[6,6],[5.5,5.5]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    return X_train, y_train


@pytest.fixture
def three_class():
    """Three separable classes in 2D."""
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0],   0.3, (20, 2))
    X1 = rng.normal([5, 0],   0.3, (20, 2))
    X2 = rng.normal([2.5, 4], 0.3, (20, 2))
    X = np.vstack([X0, X1, X2])
    y = np.array([0]*20 + [1]*20 + [2]*20)
    return X, y


@pytest.fixture
def regression_data():
    """Simple noisy linear data for regression mode."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(80, 1))
    y = 2 * X.squeeze() + rng.normal(0, 0.5, 80)
    return X, y


# ---------------------------------------------------------------------------
# Basic Classification
# ---------------------------------------------------------------------------

def test_predict_class_zero(two_cluster):
    """Should predict class 0 for a point near the origin."""
    X, y = two_cluster
    clf = KNN(k=3)
    clf.fit(X, y)
    assert clf.predict(np.array([[0.2, 0.2]]))[0] == 0


def test_predict_class_one(two_cluster):
    """Should predict class 1 for a point near [5, 5]."""
    X, y = two_cluster
    clf = KNN(k=3)
    clf.fit(X, y)
    assert clf.predict(np.array([[5.2, 5.2]]))[0] == 1


def test_majority_vote(two_cluster):
    """Majority vote should determine the class when neighbors are mixed."""
    X_train = np.array([[1,1],[1.1,1.1],[0,0]], dtype=float)
    y_train = np.array([1, 1, 0])
    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    assert clf.predict(np.array([[0.9, 0.9]]))[0] == 1


def test_perfect_accuracy_separable(two_cluster):
    """Should achieve 100% accuracy on clearly separated clusters."""
    X, y = two_cluster
    clf = KNN(k=3)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


def test_multiclass(three_class):
    """Should handle three-class classification correctly."""
    X, y = three_class
    clf = KNN(k=5)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.95


def test_k_equals_one(two_cluster):
    """k=1 should perfectly memorize training labels."""
    X, y = two_cluster
    clf = KNN(k=1)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# Regression Mode
# ---------------------------------------------------------------------------

def test_regression_r2(regression_data):
    """KNN regression R² should be high on low-noise linear data."""
    X, y = regression_data
    reg = KNN(k=5, task="regression")
    reg.fit(X, y)
    assert reg.score(X, y) > 0.90


def test_regression_output_shape(regression_data):
    """Regression predict() should return shape (n_samples,)."""
    X, y = regression_data
    reg = KNN(k=3, task="regression")
    reg.fit(X, y)
    assert reg.predict(X).shape == (X.shape[0],)


def test_regression_single_point(regression_data):
    """Regression should return a scalar-like value for a single point."""
    X, y = regression_data
    reg = KNN(k=3, task="regression")
    reg.fit(X, y)
    result = reg.predict(np.array([[5.0]]))
    assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Distance Metrics
# ---------------------------------------------------------------------------

def test_manhattan_metric(two_cluster):
    """Manhattan metric should still classify separated clusters correctly."""
    X, y = two_cluster
    clf = KNN(k=3, metric="manhattan")
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


def test_minkowski_metric(two_cluster):
    """Minkowski metric (p=3) should classify separated clusters correctly."""
    X, y = two_cluster
    clf = KNN(k=3, metric="minkowski", p=3)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# Distance Weighting
# ---------------------------------------------------------------------------

def test_distance_weighting(two_cluster):
    """Distance-weighted KNN should achieve high accuracy on separable data."""
    X, y = two_cluster
    clf = KNN(k=3, weights="distance")
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


def test_distance_weighting_regression(regression_data):
    """Distance-weighted regression should achieve reasonable R²."""
    X, y = regression_data
    reg = KNN(k=5, task="regression", weights="distance")
    reg.fit(X, y)
    assert reg.score(X, y) > 0.90


# ---------------------------------------------------------------------------
# predict() & score()
# ---------------------------------------------------------------------------

def test_predict_output_shape(two_cluster):
    """predict() should return shape (n_samples,)."""
    X, y = two_cluster
    clf = KNN(k=3)
    clf.fit(X, y)
    assert clf.predict(X).shape == (X.shape[0],)


def test_score_range(two_cluster):
    """score() should return a value in [0, 1] for classification."""
    X, y = two_cluster
    clf = KNN(k=3)
    clf.fit(X, y)
    assert 0.0 <= clf.score(X, y) <= 1.0


def test_score_matches_manual(two_cluster):
    """score() should match manually computed accuracy."""
    X, y = two_cluster
    clf = KNN(k=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert clf.score(X, y) == pytest.approx(np.mean(preds == y))


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict() before fit() should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        KNN().predict(np.array([[1, 2]]))


def test_invalid_k():
    """k <= 0 should raise ValueError."""
    with pytest.raises(ValueError):
        KNN(k=0)
    with pytest.raises(ValueError):
        KNN(k=-1)


def test_k_exceeds_samples_raises():
    """k larger than training set should raise ValueError."""
    X = np.array([[1, 2], [3, 4]], dtype=float)
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        KNN(k=5).fit(X, y)


def test_invalid_metric():
    """Unknown metric should raise ValueError."""
    with pytest.raises(ValueError):
        KNN(metric="cosine")


def test_invalid_task():
    """Unknown task should raise ValueError."""
    with pytest.raises(ValueError):
        KNN(task="clustering")


def test_invalid_weights():
    """Unknown weights should raise ValueError."""
    with pytest.raises(ValueError):
        KNN(weights="gaussian")


def test_mismatched_X_y_raises():
    """Mismatched X and y sample counts should raise ValueError."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        KNN(k=2).fit(X, y)


def test_method_chaining(two_cluster):
    """fit() should return self to support method chaining."""
    X, y = two_cluster
    model = KNN(k=3)
    assert model.fit(X, y) is model


def test_single_neighbor():
    """k=1 should return the label of the nearest training point."""
    X_train = np.array([[0, 0], [10, 10]], dtype=float)
    y_train = np.array([42, 99])
    clf = KNN(k=1)
    clf.fit(X_train, y_train)
    assert clf.predict(np.array([[0.1, 0.1]]))[0] == 42
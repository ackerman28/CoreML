import numpy as np
import pytest
from mlpackage.supervised_learning.random_forest import RandomForest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_cluster():
    X = np.array([[1,2],[2,3],[3,3],[8,8],[9,9],[10,10]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def gaussian_clusters():
    rng = np.random.default_rng(42)
    X0 = rng.normal([-3, -3], 0.5, (50, 2))
    X1 = rng.normal([ 3,  3], 0.5, (50, 2))
    X  = np.vstack([X0, X1])
    y  = np.array([0]*50 + [1]*50)
    return X, y


@pytest.fixture
def three_class():
    rng = np.random.default_rng(1)
    X0 = rng.normal([0, 0],   0.4, (30, 2))
    X1 = rng.normal([5, 0],   0.4, (30, 2))
    X2 = rng.normal([2.5, 4], 0.4, (30, 2))
    X  = np.vstack([X0, X1, X2])
    y  = np.array([0]*30 + [1]*30 + [2]*30)
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(120, 1))
    y = 3 * X.squeeze() + rng.normal(0, 0.5, 120)
    return X, y


# ---------------------------------------------------------------------------
# Basic Classification
# ---------------------------------------------------------------------------

def test_basic_accuracy(two_cluster):
    """Should achieve high accuracy on clearly separated clusters."""
    X, y = two_cluster
    rf = RandomForest(n_trees=20, random_state=42)
    rf.fit(X, y)
    assert rf.score(X, y) >= 0.83


def test_perfect_accuracy_separable(gaussian_clusters):
    """Should achieve 100% on well-separated Gaussian clusters."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=50, max_depth=5, random_state=42)
    rf.fit(X, y)
    assert rf.score(X, y) == 1.0


def test_multiclass(three_class):
    """Should handle three-class classification."""
    X, y = three_class
    rf = RandomForest(n_trees=30, max_depth=5, random_state=42)
    rf.fit(X, y)
    assert rf.score(X, y) > 0.95


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def test_regression_r2(regression_data):
    """Regression forest should achieve high R² on low-noise data."""
    X, y = regression_data
    rf = RandomForest(n_trees=50, max_depth=5, task="regression", random_state=42)
    rf.fit(X, y)
    assert rf.score(X, y) > 0.90


def test_regression_output_shape(regression_data):
    """Regression predict() should return shape (n_samples,)."""
    X, y = regression_data
    rf = RandomForest(n_trees=10, task="regression", random_state=42)
    rf.fit(X, y)
    assert rf.predict(X).shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# OOB Score
# ---------------------------------------------------------------------------

def test_oob_score_computed(gaussian_clusters):
    """oob_score_ should be set when oob_score=True."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=50, random_state=42)
    rf.fit(X, y, oob_score=True)
    assert rf.oob_score_ is not None
    assert 0.0 <= rf.oob_score_ <= 1.0


def test_oob_score_not_set_by_default(gaussian_clusters):
    """oob_score_ should be None when oob_score=False."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=20, random_state=42)
    rf.fit(X, y, oob_score=False)
    assert rf.oob_score_ is None


def test_oob_score_reasonable(gaussian_clusters):
    """OOB score should be a reasonable estimate of generalization."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=100, random_state=42)
    rf.fit(X, y, oob_score=True)
    assert rf.oob_score_ > 0.85


# ---------------------------------------------------------------------------
# Feature Importances
# ---------------------------------------------------------------------------

def test_feature_importances_shape(gaussian_clusters):
    """feature_importances_ should have one entry per feature."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=20, random_state=42)
    rf.fit(X, y)
    assert rf.feature_importances_.shape == (X.shape[1],)


def test_feature_importances_sum_to_one(gaussian_clusters):
    """Feature importances should sum to approximately 1.0."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=20, random_state=42)
    rf.fit(X, y)
    assert np.isclose(rf.feature_importances_.sum(), 1.0, atol=1e-6)


def test_feature_importances_nonnegative(gaussian_clusters):
    """All feature importances should be non-negative."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=20, random_state=42)
    rf.fit(X, y)
    assert np.all(rf.feature_importances_ >= 0)


# ---------------------------------------------------------------------------
# predict_proba()
# ---------------------------------------------------------------------------

def test_predict_proba_shape(gaussian_clusters):
    """predict_proba() should return shape (n_samples, n_classes)."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=20, random_state=42)
    rf.fit(X, y)
    proba = rf.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert proba.shape[1] == 2


def test_predict_proba_sums_to_one(gaussian_clusters):
    """Each row of predict_proba() should sum to 1."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=20, random_state=42)
    rf.fit(X, y)
    proba = rf.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_proba_regression_raises(regression_data):
    """predict_proba() should raise RuntimeError for regression task."""
    X, y = regression_data
    rf = RandomForest(n_trees=10, task="regression", random_state=42)
    rf.fit(X, y)
    with pytest.raises(RuntimeError):
        rf.predict_proba(X)


# ---------------------------------------------------------------------------
# predict() & score()
# ---------------------------------------------------------------------------

def test_predict_output_shape(gaussian_clusters):
    """predict() should return shape (n_samples,)."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=10, random_state=42)
    rf.fit(X, y)
    assert rf.predict(X).shape == (X.shape[0],)


def test_score_range(gaussian_clusters):
    """score() should return a value in [0, 1] for classification."""
    X, y = gaussian_clusters
    rf = RandomForest(n_trees=10, random_state=42)
    rf.fit(X, y)
    assert 0.0 <= rf.score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict() before fit() should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        RandomForest().predict(np.array([[1, 2]]))


def test_invalid_n_trees():
    with pytest.raises(ValueError):
        RandomForest(n_trees=0)


def test_invalid_max_depth():
    with pytest.raises(ValueError):
        RandomForest(max_depth=0)


def test_invalid_criterion():
    with pytest.raises(ValueError):
        RandomForest(criterion="log_loss")


def test_invalid_task():
    with pytest.raises(ValueError):
        RandomForest(task="clustering")


def test_mismatched_X_y_raises():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        RandomForest().fit(X, y)


def test_method_chaining(two_cluster):
    """fit() should return self to support method chaining."""
    X, y = two_cluster
    rf = RandomForest(n_trees=5, random_state=0)
    assert rf.fit(X, y) is rf


def test_reproducibility(gaussian_clusters):
    """Same random_state should produce identical predictions."""
    X, y = gaussian_clusters
    rf1 = RandomForest(n_trees=20, random_state=7)
    rf2 = RandomForest(n_trees=20, random_state=7)
    rf1.fit(X, y)
    rf2.fit(X, y)
    assert np.array_equal(rf1.predict(X), rf2.predict(X))
import numpy as np
import pytest
from mlpackage.supervised_learning.decision_tree import DecisionTree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_xor():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    y = np.array([0, 1, 0, 1])
    return X, y


@pytest.fixture
def separable_2d():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 0.4, (30, 2))
    X1 = rng.normal([4, 4], 0.4, (30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0]*30 + [1]*30)
    return X, y


@pytest.fixture
def three_class():
    rng = np.random.default_rng(1)
    X0 = rng.normal([0, 0],   0.3, (20, 2))
    X1 = rng.normal([5, 0],   0.3, (20, 2))
    X2 = rng.normal([2.5, 4], 0.3, (20, 2))
    X = np.vstack([X0, X1, X2])
    y = np.array([0]*20 + [1]*20 + [2]*20)
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(100, 1))
    y = 2 * X.squeeze() + rng.normal(0, 0.5, 100)
    return X, y


# ---------------------------------------------------------------------------
# Basic Classification
# ---------------------------------------------------------------------------

def test_memorizes_training_data(simple_xor):
    """Unconstrained tree should perfectly memorize training labels."""
    X, y = simple_xor
    clf = DecisionTree(max_depth=10)
    clf.fit(X, y)
    assert np.array_equal(clf.predict(X), y)


def test_perfect_accuracy_separable(separable_2d):
    """Should achieve 100% accuracy on clearly separated clusters."""
    X, y = separable_2d
    clf = DecisionTree(max_depth=10)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


def test_multiclass(three_class):
    """Should handle three-class classification correctly."""
    X, y = three_class
    clf = DecisionTree(max_depth=10)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.95


def test_shallow_tree_underfits(separable_2d):
    """A tree of max_depth=1 (stump) should have lower accuracy than deep tree."""
    X, y = separable_2d
    stump = DecisionTree(max_depth=1)
    deep  = DecisionTree(max_depth=10)
    stump.fit(X, y)
    deep.fit(X, y)
    assert deep.score(X, y) >= stump.score(X, y)


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------

def test_gini_criterion(separable_2d):
    """Gini criterion should also achieve high accuracy on separable data."""
    X, y = separable_2d
    clf = DecisionTree(criterion="gini", max_depth=10)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


def test_entropy_and_gini_agree(separable_2d):
    """Entropy and Gini should produce the same predictions on clean data."""
    X, y = separable_2d
    clf_e = DecisionTree(criterion="entropy", max_depth=10, random_state=0)
    clf_g = DecisionTree(criterion="gini",    max_depth=10, random_state=0)
    clf_e.fit(X, y)
    clf_g.fit(X, y)
    assert np.array_equal(clf_e.predict(X), clf_g.predict(X))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def test_regression_r2(regression_data):
    """Regression tree R² should be high on low-noise data."""
    X, y = regression_data
    reg = DecisionTree(task="regression", max_depth=10)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.90


def test_regression_output_shape(regression_data):
    """Regression predict() should return shape (n_samples,)."""
    X, y = regression_data
    reg = DecisionTree(task="regression", max_depth=5)
    reg.fit(X, y)
    assert reg.predict(X).shape == (X.shape[0],)


def test_regression_deeper_is_better(regression_data):
    """Deeper regression tree should have higher R² than shallow one."""
    X, y = regression_data
    shallow = DecisionTree(task="regression", max_depth=2)
    deep    = DecisionTree(task="regression", max_depth=10)
    shallow.fit(X, y)
    deep.fit(X, y)
    assert deep.score(X, y) >= shallow.score(X, y)


# ---------------------------------------------------------------------------
# Feature Importances
# ---------------------------------------------------------------------------

def test_feature_importances_shape(separable_2d):
    """feature_importances_ should have one entry per feature."""
    X, y = separable_2d
    clf = DecisionTree(max_depth=5)
    clf.fit(X, y)
    assert clf.feature_importances_.shape == (X.shape[1],)


def test_feature_importances_sum_to_one(separable_2d):
    """Feature importances should sum to 1.0."""
    X, y = separable_2d
    clf = DecisionTree(max_depth=5)
    clf.fit(X, y)
    assert np.isclose(clf.feature_importances_.sum(), 1.0)


def test_feature_importances_nonnegative(separable_2d):
    """All feature importances should be non-negative."""
    X, y = separable_2d
    clf = DecisionTree(max_depth=5)
    clf.fit(X, y)
    assert np.all(clf.feature_importances_ >= 0)


# ---------------------------------------------------------------------------
# predict() & score()
# ---------------------------------------------------------------------------

def test_predict_output_shape(separable_2d):
    """predict() should return shape (n_samples,)."""
    X, y = separable_2d
    clf = DecisionTree()
    clf.fit(X, y)
    assert clf.predict(X).shape == (X.shape[0],)


def test_score_range(separable_2d):
    """score() should return a value in [0, 1] for classification."""
    X, y = separable_2d
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)
    assert 0.0 <= clf.score(X, y) <= 1.0


def test_score_matches_manual(separable_2d):
    """score() should match manually computed accuracy."""
    X, y = separable_2d
    clf = DecisionTree(max_depth=5)
    clf.fit(X, y)
    assert clf.score(X, y) == pytest.approx(np.mean(clf.predict(X) == y))


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict() before fit() should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        DecisionTree().predict(np.array([[1, 2]]))


def test_invalid_min_samples_split():
    """min_samples_split < 2 should raise ValueError."""
    with pytest.raises(ValueError):
        DecisionTree(min_samples_split=1)


def test_invalid_max_depth():
    """max_depth < 1 should raise ValueError."""
    with pytest.raises(ValueError):
        DecisionTree(max_depth=0)


def test_invalid_criterion():
    """Unknown criterion should raise ValueError."""
    with pytest.raises(ValueError):
        DecisionTree(criterion="log_loss")


def test_invalid_task():
    """Unknown task should raise ValueError."""
    with pytest.raises(ValueError):
        DecisionTree(task="clustering")


def test_mismatched_X_y_raises():
    """Mismatched X and y sample counts should raise ValueError."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        DecisionTree().fit(X, y)


def test_single_class_input():
    """All-same-label data should produce a single leaf without error."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    y = np.array([1, 1, 1])
    clf = DecisionTree()
    clf.fit(X, y)
    assert np.all(clf.predict(X) == 1)


def test_method_chaining(separable_2d):
    """fit() should return self to support method chaining."""
    X, y = separable_2d
    model = DecisionTree()
    assert model.fit(X, y) is model


def test_reproducibility(separable_2d):
    """Same random_state should produce identical predictions."""
    X, y = separable_2d
    clf1 = DecisionTree(n_features=1, random_state=7)
    clf2 = DecisionTree(n_features=1, random_state=7)
    clf1.fit(X, y)
    clf2.fit(X, y)
    assert np.array_equal(clf1.predict(X), clf2.predict(X))
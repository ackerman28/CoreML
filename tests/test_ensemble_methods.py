import numpy as np
import pytest
from mlpackage.supervised_learning.ensemble_methods import (
    _BaseBagging, BaggingClassifier, BaggingRegressor, VotingClassifier
)
from mlpackage.supervised_learning.decision_tree import DecisionTree
from mlpackage.supervised_learning.knn import KNN
from mlpackage.supervised_learning.perceptron import Perceptron


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    X = np.array([[1,2],[2,3],[3,3],[8,8],[9,9],[10,10]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def gaussian_clusters():
    rng = np.random.default_rng(42)
    X0 = rng.normal([-3,-3], 0.5, (40, 2))
    X1 = rng.normal([ 3, 3], 0.5, (40, 2))
    X  = np.vstack([X0, X1])
    y  = np.array([0]*40 + [1]*40)
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(80, 1))
    y = 2.5 * X.squeeze() + rng.normal(0, 0.5, 80)
    return X, y


# ---------------------------------------------------------------------------
# _BaseBagging
# ---------------------------------------------------------------------------

def test_base_bagging_with_perceptron(binary_data):
    """Bagging a Perceptron should produce correct predictions on separable data."""
    X, y = binary_data
    ens = _BaseBagging(Perceptron(learning_rate=0.1, n_iters=300),
                       n_estimators=5, random_state=42, mode="hard_vote")
    ens.fit(X, y)
    assert np.array_equal(ens.predict(X), y)


def test_base_bagging_n_estimators_stored(gaussian_clusters):
    """estimators_ should contain exactly n_estimators fitted models."""
    X, y = gaussian_clusters
    ens = _BaseBagging(DecisionTree(max_depth=3), n_estimators=7, random_state=0)
    ens.fit(X, y)
    assert len(ens.estimators_) == 7


def test_base_bagging_average_mode(regression_data):
    """Bagging in average mode should return float predictions."""
    X, y = regression_data
    ens = _BaseBagging(DecisionTree(task="regression", max_depth=3),
                       n_estimators=5, random_state=0, mode="average")
    ens.fit(X, y)
    preds = ens.predict(X)
    assert preds.shape == (X.shape[0],)


def test_base_bagging_predict_before_fit_raises():
    """predict() before fit() should raise RuntimeError."""
    ens = _BaseBagging(DecisionTree(), n_estimators=5)
    with pytest.raises(RuntimeError):
        ens.predict(np.ones((3, 2)))


def test_base_bagging_invalid_n_estimators():
    with pytest.raises(ValueError):
        _BaseBagging(DecisionTree(), n_estimators=0)


def test_base_bagging_invalid_mode():
    with pytest.raises(ValueError):
        _BaseBagging(DecisionTree(), mode="soft_vote")


# ---------------------------------------------------------------------------
# BaggingClassifier
# ---------------------------------------------------------------------------

def test_bagging_classifier_accuracy(gaussian_clusters):
    """BaggingClassifier should achieve high accuracy on separable data."""
    X, y = gaussian_clusters
    clf = BaggingClassifier(DecisionTree(max_depth=5), n_estimators=20, random_state=42)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.95


def test_bagging_classifier_predict_shape(gaussian_clusters):
    """predict() should return shape (n_samples,)."""
    X, y = gaussian_clusters
    clf = BaggingClassifier(DecisionTree(max_depth=3), n_estimators=5, random_state=0)
    clf.fit(X, y)
    assert clf.predict(X).shape == (X.shape[0],)


def test_bagging_classifier_predict_proba_shape(gaussian_clusters):
    """predict_proba() should return shape (n_samples, n_classes)."""
    X, y = gaussian_clusters
    clf = BaggingClassifier(DecisionTree(max_depth=3), n_estimators=10, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert proba.shape[1] == 2


def test_bagging_classifier_proba_sums_to_one(gaussian_clusters):
    """Each row of predict_proba() should sum to 1."""
    X, y = gaussian_clusters
    clf = BaggingClassifier(DecisionTree(max_depth=3), n_estimators=10, random_state=0)
    clf.fit(X, y)
    assert np.allclose(clf.predict_proba(X).sum(axis=1), 1.0)


def test_bagging_classifier_reproducibility(gaussian_clusters):
    """Same random_state should produce identical predictions."""
    X, y = gaussian_clusters
    clf1 = BaggingClassifier(DecisionTree(max_depth=3), n_estimators=10, random_state=7)
    clf2 = BaggingClassifier(DecisionTree(max_depth=3), n_estimators=10, random_state=7)
    clf1.fit(X, y)
    clf2.fit(X, y)
    assert np.array_equal(clf1.predict(X), clf2.predict(X))


# ---------------------------------------------------------------------------
# BaggingRegressor
# ---------------------------------------------------------------------------

def test_bagging_regressor_r2(regression_data):
    """BaggingRegressor R² should be high on low-noise data."""
    X, y = regression_data
    reg = BaggingRegressor(DecisionTree(task="regression", max_depth=5),
                           n_estimators=20, random_state=42)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.90


def test_bagging_regressor_output_shape(regression_data):
    """Regression predict() should return shape (n_samples,)."""
    X, y = regression_data
    reg = BaggingRegressor(DecisionTree(task="regression", max_depth=3),
                           n_estimators=5, random_state=0)
    reg.fit(X, y)
    assert reg.predict(X).shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# VotingClassifier
# ---------------------------------------------------------------------------

def test_voting_hard_accuracy(gaussian_clusters):
    """Hard voting should achieve high accuracy on separable data."""
    X, y = gaussian_clusters
    clf = VotingClassifier([
        ("dt",  DecisionTree(max_depth=5, random_state=0)),
        ("knn", KNN(k=5)),
    ], voting="hard")
    clf.fit(X, y)
    assert clf.score(X, y) > 0.95


def test_voting_predict_shape(gaussian_clusters):
    """predict() should return shape (n_samples,)."""
    X, y = gaussian_clusters
    clf = VotingClassifier([
        ("dt",  DecisionTree(max_depth=3, random_state=0)),
        ("knn", KNN(k=3)),
    ])
    clf.fit(X, y)
    assert clf.predict(X).shape == (X.shape[0],)


def test_voting_invalid_mode():
    with pytest.raises(ValueError):
        VotingClassifier([("dt", DecisionTree())], voting="medium")


def test_voting_empty_estimators():
    with pytest.raises(ValueError):
        VotingClassifier([])


def test_voting_predict_before_fit_raises():
    clf = VotingClassifier([("dt", DecisionTree())])
    with pytest.raises(RuntimeError):
        clf.predict(np.ones((3, 2)))


def test_voting_score_range(gaussian_clusters):
    """score() should return a value in [0, 1]."""
    X, y = gaussian_clusters
    clf = VotingClassifier([
        ("dt",  DecisionTree(max_depth=3, random_state=0)),
        ("knn", KNN(k=5)),
    ])
    clf.fit(X, y)
    assert 0.0 <= clf.score(X, y) <= 1.0
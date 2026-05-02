import numpy as np
import pytest
from mlpackage.unsupervised_learning.dbscan import DBSCAN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_clusters_with_noise():
    """Two tight clusters and one clear outlier."""
    X = np.array([
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0],
        [5.0, 5.0], [5.1, 5.1], [5.2, 5.0],
        [10.0, 10.0]
    ])
    return X


@pytest.fixture
def three_clusters():
    rng = np.random.default_rng(42)
    X0 = rng.normal([0, 0],   0.2, (20, 2))
    X1 = rng.normal([5, 0],   0.2, (20, 2))
    X2 = rng.normal([2.5, 4], 0.2, (20, 2))
    return np.vstack([X0, X1, X2])


# ---------------------------------------------------------------------------
# Core Clustering Behavior
# ---------------------------------------------------------------------------

def test_two_clusters_detected(two_clusters_with_noise):
    """Should detect exactly 2 clusters in simple two-blob data."""
    db = DBSCAN(eps=1.0, min_samples=2)
    db.fit(two_clusters_with_noise)
    assert db.n_clusters_ == 2


def test_noise_point_labeled_minus_one(two_clusters_with_noise):
    """The outlier point should be labeled -1."""
    db = DBSCAN(eps=1.0, min_samples=2)
    db.fit(two_clusters_with_noise)
    assert db.labels_[-1] == -1


def test_cluster_labels_present(two_clusters_with_noise):
    """Labels 0 and 1 should both appear in the cluster assignments."""
    db = DBSCAN(eps=1.0, min_samples=2)
    db.fit(two_clusters_with_noise)
    unique = set(db.labels_)
    assert 0 in unique
    assert 1 in unique
    assert -1 in unique


def test_same_blob_same_label(two_clusters_with_noise):
    """Points in the same blob should share a cluster label."""
    db = DBSCAN(eps=1.0, min_samples=2)
    db.fit(two_clusters_with_noise)
    assert db.labels_[0] == db.labels_[1] == db.labels_[2]
    assert db.labels_[3] == db.labels_[4] == db.labels_[5]


def test_different_blobs_different_labels(two_clusters_with_noise):
    """Points from different blobs should have different cluster labels."""
    db = DBSCAN(eps=1.0, min_samples=2)
    db.fit(two_clusters_with_noise)
    assert db.labels_[0] != db.labels_[3]


def test_three_clusters(three_clusters):
    """Should detect exactly 3 clusters on well-separated Gaussian blobs."""
    db = DBSCAN(eps=0.5, min_samples=3)
    db.fit(three_clusters)
    assert db.n_clusters_ == 3


# ---------------------------------------------------------------------------
# Noise & Core Points
# ---------------------------------------------------------------------------

def test_n_noise_counted(two_clusters_with_noise):
    """n_noise_ should equal number of points labeled -1."""
    db = DBSCAN(eps=1.0, min_samples=2)
    db.fit(two_clusters_with_noise)
    assert db.n_noise_ == int(np.sum(db.labels_ == -1))


def test_core_sample_indices_set(two_clusters_with_noise):
    """core_sample_indices_ should be set and non-empty after fitting."""
    db = DBSCAN(eps=1.0, min_samples=2)
    db.fit(two_clusters_with_noise)
    assert db.core_sample_indices_ is not None
    assert len(db.core_sample_indices_) > 0


def test_high_eps_one_cluster():
    """Very large eps should merge everything into a single cluster."""
    X = np.array([[0,0],[1,1],[2,2],[10,10]], dtype=float)
    db = DBSCAN(eps=100, min_samples=2)
    db.fit(X)
    assert db.n_clusters_ == 1
    assert db.n_noise_ == 0


def test_high_min_samples_all_noise():
    """Very high min_samples should label all points as noise."""
    X = np.array([[0,0],[1,1],[2,2]], dtype=float)
    db = DBSCAN(eps=1.5, min_samples=100)
    db.fit(X)
    assert np.all(db.labels_ == -1)
    assert db.n_clusters_ == 0


# ---------------------------------------------------------------------------
# Labels Shape & Attributes
# ---------------------------------------------------------------------------

def test_labels_shape(three_clusters):
    """labels_ should have shape (n_samples,)."""
    db = DBSCAN(eps=0.5, min_samples=3)
    db.fit(three_clusters)
    assert db.labels_.shape == (three_clusters.shape[0],)


def test_n_clusters_attribute(three_clusters):
    """n_clusters_ should match number of unique non-noise labels."""
    db = DBSCAN(eps=0.5, min_samples=3)
    db.fit(three_clusters)
    expected = len(set(db.labels_) - {-1})
    assert db.n_clusters_ == expected


def test_fit_predict_matches_labels(two_clusters_with_noise):
    """fit_predict() should return same result as fit().labels_."""
    db1 = DBSCAN(eps=1.0, min_samples=2)
    db2 = DBSCAN(eps=1.0, min_samples=2)
    labels1 = db1.fit_predict(two_clusters_with_noise)
    db2.fit(two_clusters_with_noise)
    assert np.array_equal(labels1, db2.labels_)


# ---------------------------------------------------------------------------
# Distance Metric
# ---------------------------------------------------------------------------

def test_manhattan_metric(two_clusters_with_noise):
    """Manhattan metric should also find 2 clusters on clear data."""
    db = DBSCAN(eps=1.5, min_samples=2, metric="manhattan")
    db.fit(two_clusters_with_noise)
    assert db.n_clusters_ == 2


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

def test_invalid_eps():
    with pytest.raises(ValueError):
        DBSCAN(eps=0)
    with pytest.raises(ValueError):
        DBSCAN(eps=-1.0)


def test_invalid_min_samples():
    with pytest.raises(ValueError):
        DBSCAN(min_samples=0)


def test_invalid_metric():
    with pytest.raises(ValueError):
        DBSCAN(metric="cosine")


def test_non_2d_input_raises():
    """1D input should raise ValueError."""
    db = DBSCAN()
    with pytest.raises(ValueError):
        db.fit(np.array([1, 2, 3]))


def test_method_chaining():
    """fit() should return self."""
    X = np.array([[0,0],[0.1,0],[5,5]], dtype=float)
    db = DBSCAN(eps=0.5, min_samples=2)
    assert db.fit(X) is db
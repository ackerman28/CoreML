import numpy as np
import pytest
from mlpackage.unsupervised_learning.kmeans import KMeans


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blobs():
    """Two clearly separated blobs."""
    X = np.array([[1,1],[1.1,1.1],[0.9,0.9],[10,10],[10.1,10.1],[9.9,9.9]], dtype=float)
    return X


@pytest.fixture
def three_blobs():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0],   0.3, (30, 2))
    X1 = rng.normal([6, 0],   0.3, (30, 2))
    X2 = rng.normal([3, 5],   0.3, (30, 2))
    return np.vstack([X0, X1, X2])


# ---------------------------------------------------------------------------
# Cluster Assignment Correctness
# ---------------------------------------------------------------------------

def test_two_blobs_same_label(two_blobs):
    """Points in the same blob should receive the same cluster label."""
    km = KMeans(k=2, random_state=42)
    km.fit(two_blobs)
    labels = km.labels_
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_three_blobs_cluster_count(three_blobs):
    """Should find exactly k=3 distinct clusters."""
    km = KMeans(k=3, random_state=42)
    km.fit(three_blobs)
    assert len(np.unique(km.labels_)) == 3


def test_predict_consistent_with_labels(two_blobs):
    """predict() on training data should match labels_ attribute."""
    km = KMeans(k=2, random_state=42)
    km.fit(two_blobs)
    assert np.array_equal(km.predict(two_blobs), km.labels_)


def test_fit_predict_matches_fit_then_predict(three_blobs):
    """fit_predict() should return same labels as fit() then labels_."""
    km1 = KMeans(k=3, random_state=0)
    km2 = KMeans(k=3, random_state=0)
    labels1 = km1.fit_predict(three_blobs)
    km2.fit(three_blobs)
    assert np.array_equal(labels1, km2.labels_)


# ---------------------------------------------------------------------------
# Inertia
# ---------------------------------------------------------------------------

def test_inertia_positive(two_blobs):
    """Inertia should be non-negative after fitting."""
    km = KMeans(k=2, random_state=42)
    km.fit(two_blobs)
    assert km.inertia_ >= 0


def test_more_clusters_lower_inertia(three_blobs):
    """Inertia should decrease as k increases."""
    inertias = []
    for k in [1, 2, 3, 4]:
        km = KMeans(k=k, random_state=42)
        km.fit(three_blobs)
        inertias.append(km.inertia_)
    assert all(inertias[i] >= inertias[i+1] for i in range(len(inertias)-1))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_random_init(two_blobs):
    """random init should still find the correct clusters on clear data."""
    km = KMeans(k=2, init="random", random_state=42)
    km.fit(two_blobs)
    assert km.labels_[0] == km.labels_[1]
    assert km.labels_[0] != km.labels_[3]


def test_kmeans_plus_plus_init(two_blobs):
    """kmeans++ init should find correct clusters."""
    km = KMeans(k=2, init="kmeans++", random_state=42)
    km.fit(two_blobs)
    assert km.labels_[0] == km.labels_[1]
    assert km.labels_[0] != km.labels_[3]


def test_reproducibility(three_blobs):
    """Same random_state should produce identical labels."""
    km1 = KMeans(k=3, random_state=7)
    km2 = KMeans(k=3, random_state=7)
    km1.fit(three_blobs)
    km2.fit(three_blobs)
    assert np.array_equal(km1.labels_, km2.labels_)


# ---------------------------------------------------------------------------
# Centroids & Attributes
# ---------------------------------------------------------------------------

def test_centroids_shape(three_blobs):
    """centroids_ should have shape (k, n_features)."""
    km = KMeans(k=3, random_state=0)
    km.fit(three_blobs)
    assert km.centroids_.shape == (3, 2)


def test_n_iter_recorded(three_blobs):
    """n_iter_ should be set and positive after fitting."""
    km = KMeans(k=3, random_state=0)
    km.fit(three_blobs)
    assert km.n_iter_ is not None
    assert km.n_iter_ > 0


def test_labels_shape(three_blobs):
    """labels_ should have shape (n_samples,)."""
    km = KMeans(k=3, random_state=0)
    km.fit(three_blobs)
    assert km.labels_.shape == (three_blobs.shape[0],)


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict() before fit() should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        KMeans(k=2).predict(np.ones((4, 2)))


def test_invalid_k():
    with pytest.raises(ValueError):
        KMeans(k=0)


def test_invalid_init():
    with pytest.raises(ValueError):
        KMeans(init="bad")


def test_invalid_max_iters():
    with pytest.raises(ValueError):
        KMeans(max_iters=0)


def test_k_exceeds_samples_raises():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    with pytest.raises(ValueError):
        KMeans(k=5).fit(X)


def test_method_chaining(two_blobs):
    """fit() should return self."""
    km = KMeans(k=2, random_state=0)
    assert km.fit(two_blobs) is km
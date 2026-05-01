import numpy as np
import pytest
from mlpackage.unsupervised_learning.pca import PCA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 5))


@pytest.fixture
def correlated_data():
    """Data lying approximately along a line in 2D."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 100)
    X = np.column_stack([t + rng.normal(0, 0.1, 100),
                         2*t + rng.normal(0, 0.1, 100)])
    return X


@pytest.fixture
def line_data():
    """Exact line y=x — variance fully captured by PC1."""
    return np.array([[1,1],[2,2],[3,3],[4,4],[5,5]], dtype=float)


# ---------------------------------------------------------------------------
# Output Shape
# ---------------------------------------------------------------------------

def test_transform_shape(random_data):
    """Transformed output should have shape (n_samples, n_components)."""
    pca = PCA(n_components=2)
    pca.fit(random_data)
    assert pca.transform(random_data).shape == (50, 2)


def test_fit_transform_shape(random_data):
    """fit_transform() should return same shape as fit + transform."""
    pca = PCA(n_components=3)
    out = pca.fit_transform(random_data)
    assert out.shape == (50, 3)


def test_components_shape(random_data):
    """components_ should have shape (n_components, n_features)."""
    pca = PCA(n_components=3)
    pca.fit(random_data)
    assert pca.components_.shape == (3, 5)


def test_inverse_transform_shape(random_data):
    """inverse_transform() should recover original feature dimensionality."""
    pca = PCA(n_components=3)
    X_proj = pca.fit_transform(random_data)
    X_rec  = pca.inverse_transform(X_proj)
    assert X_rec.shape == random_data.shape


# ---------------------------------------------------------------------------
# Mathematical Properties
# ---------------------------------------------------------------------------

def test_mean_centering(random_data):
    """Transformed data should have approximately zero mean."""
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(random_data)
    assert np.allclose(np.mean(X_proj, axis=0), 0, atol=1e-10)


def test_components_orthogonal(random_data):
    """Principal components should be mutually orthogonal."""
    pca = PCA(n_components=4)
    pca.fit(random_data)
    gram = pca.components_ @ pca.components_.T
    assert np.allclose(gram, np.eye(4), atol=1e-10)


def test_components_unit_norm(random_data):
    """Each principal component should have unit norm."""
    pca = PCA(n_components=3)
    pca.fit(random_data)
    norms = np.linalg.norm(pca.components_, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_explained_variance_ratio_sums_to_one(random_data):
    """Explained variance ratio for all components should sum to 1."""
    pca = PCA(n_components=5)
    pca.fit(random_data)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-6)


def test_explained_variance_decreasing(random_data):
    """Eigenvalues should be in descending order."""
    pca = PCA(n_components=5)
    pca.fit(random_data)
    ev = pca.explained_variance_
    assert np.all(ev[:-1] >= ev[1:])


def test_explained_variance_ratio_range(random_data):
    """All explained variance ratios should be in [0, 1]."""
    pca = PCA(n_components=3)
    pca.fit(random_data)
    assert np.all(pca.explained_variance_ratio_ >= 0)
    assert np.all(pca.explained_variance_ratio_ <= 1)


def test_reconstruction_error_perfect_on_full_components(random_data):
    """Reconstruction with all components should recover original data exactly."""
    pca = PCA(n_components=5)
    X_proj = pca.fit_transform(random_data)
    X_rec  = pca.inverse_transform(X_proj)
    assert np.allclose(X_rec, random_data, atol=1e-8)


def test_pc1_captures_most_variance_correlated(correlated_data):
    """PC1 should capture >95% of variance in highly correlated data."""
    pca = PCA(n_components=2)
    pca.fit(correlated_data)
    assert pca.explained_variance_ratio_[0] > 0.95


def test_line_data_one_component_sufficient(line_data):
    """Data on a line should be captured almost entirely by 1 component."""
    pca = PCA(n_components=1)
    pca.fit(line_data)
    assert pca.explained_variance_ratio_[0] > 0.99


def test_fit_transform_equals_fit_then_transform(random_data):
    """fit_transform() should give same result as fit() then transform()."""
    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=2)
    out1 = pca1.fit_transform(random_data)
    pca2.fit(random_data)
    out2 = pca2.transform(random_data)
    assert np.allclose(np.abs(out1), np.abs(out2), atol=1e-10)


# ---------------------------------------------------------------------------
# Input Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_transform_before_fit_raises():
    """transform() before fit() should raise RuntimeError."""
    pca = PCA(n_components=2)
    with pytest.raises(RuntimeError):
        pca.transform(np.ones((5, 3)))


def test_inverse_transform_before_fit_raises():
    """inverse_transform() before fit() should raise RuntimeError."""
    pca = PCA(n_components=2)
    with pytest.raises(RuntimeError):
        pca.inverse_transform(np.ones((5, 2)))


def test_invalid_n_components():
    """n_components <= 0 should raise ValueError."""
    with pytest.raises(ValueError):
        PCA(n_components=0)
    with pytest.raises(ValueError):
        PCA(n_components=-1)


def test_n_components_exceeds_features_raises():
    """n_components > n_features should raise ValueError."""
    X = np.ones((10, 3))
    with pytest.raises(ValueError):
        PCA(n_components=5).fit(X)


def test_method_chaining(random_data):
    """fit() should return self to support method chaining."""
    pca = PCA(n_components=2)
    assert pca.fit(random_data) is pca
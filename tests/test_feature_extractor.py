"""
Test suite for feature_extractor.py module.

Tests NMF feature extraction and component handling.
"""
import pytest
import numpy as np
from feature_extractor import (
    extract_nmf_features,
    compute_features,
    nmf_multiplicative,
    NMFFeatureExtractor,
)


class TestNMFMultiplicative:
    """Test custom NMF multiplicative update implementation."""

    def test_nmf_multiplicative_convergence(self):
        """Test that custom NMF converges."""
        np.random.seed(42)
        V = np.random.rand(20, 15) + 0.1  # Ensure non-negative
        k = 3
        
        W, H, errors = nmf_multiplicative(V, k, max_iter=100)
        
        assert W.shape == (20, k)
        assert H.shape == (k, 15)
        assert len(errors) > 0
        # Error should be decreasing
        assert errors[-1] <= errors[0]

    def test_nmf_multiplicative_shapes(self):
        """Test NMF output shapes."""
        np.random.seed(42)
        V = np.random.rand(32, 16) + 0.1
        k = 4
        
        W, H, errors = nmf_multiplicative(V, k, max_iter=50)
        
        assert W.shape == (V.shape[0], k)
        assert H.shape == (k, V.shape[1])

    def test_nmf_multiplicative_non_negative(self):
        """Test that NMF outputs are non-negative."""
        np.random.seed(42)
        V = np.random.rand(20, 15) + 0.1
        k = 3
        
        W, H, errors = nmf_multiplicative(V, k, max_iter=100)
        
        assert np.all(W >= 0)
        assert np.all(H >= 0)


class TestExtractNMFFeatures:
    """Test NMF feature extraction function."""

    def test_extract_nmf_features_sklearn(self, mel_spectrogram):
        """Test NMF feature extraction with sklearn."""
        n_components = 4
        W, H, errors = extract_nmf_features(
            mel_spectrogram,
            n_components=n_components,
            use_sklearn=True,
        )
        
        assert W.shape[1] == n_components
        assert H.shape[0] == n_components
        assert W.shape[0] == mel_spectrogram.shape[0]
        assert H.shape[1] == mel_spectrogram.shape[1]

    def test_extract_nmf_features_custom(self, mel_spectrogram):
        """Test NMF feature extraction with custom multiplicative update."""
        n_components = 4
        W, H, errors = extract_nmf_features(
            mel_spectrogram,
            n_components=n_components,
            use_sklearn=False,
        )
        
        assert W.shape[1] == n_components
        assert H.shape[0] == n_components
        assert len(errors) > 0

    @pytest.mark.parametrize("n_components", [1, 3, 6, 8])
    def test_extract_nmf_features_different_components(self, mel_spectrogram, n_components):
        """Test NMF extraction with different component counts."""
        W, H, errors = extract_nmf_features(
            mel_spectrogram,
            n_components=n_components,
            use_sklearn=True,
        )
        
        assert W.shape[1] == n_components
        assert H.shape[0] == n_components

    def test_extract_nmf_features_non_negative(self, mel_spectrogram):
        """Test that NMF features are non-negative."""
        W, H, errors = extract_nmf_features(
            mel_spectrogram,
            n_components=4,
            use_sklearn=True,
        )
        
        assert np.all(W >= 0)
        assert np.all(H >= 0)


class TestComputeFeatures:
    """Test statistical feature computation."""

    def test_feature_vector_length(self):
        """Test that feature vector has correct length."""
        np.random.seed(42)
        n_components = 6
        W = np.random.rand(20, n_components)
        H = np.random.rand(n_components, 15)
        
        features = compute_features(W, H)
        
        # Should be n_components * 8 (4 stats from W, 4 from H)
        assert len(features) == n_components * 8

    def test_feature_vector_normalized(self):
        """Test that feature vector is normalized to [0, 1]."""
        np.random.seed(42)
        W = np.random.rand(20, 6) * 100  # Large values
        H = np.random.rand(6, 15) * 100
        
        features = compute_features(W, H)
        
        # Should be in [0, 1] range
        assert np.all(features >= 0)
        assert np.all(features <= 1)

    def test_feature_vector_dtype(self):
        """Test that feature vector is float32."""
        np.random.seed(42)
        W = np.random.rand(20, 6)
        H = np.random.rand(6, 15)
        
        features = compute_features(W, H)
        
        assert features.dtype == np.float32

    def test_compute_features_single_component(self):
        """Test feature computation with single component."""
        np.random.seed(42)
        W = np.random.rand(20, 1)
        H = np.random.rand(1, 15)
        
        features = compute_features(W, H)
        
        assert len(features) == 1 * 8
        assert np.all(features >= 0)
        assert np.all(features <= 1)

    def test_compute_features_identical_values(self):
        """Test feature computation when W and H are constant."""
        # All values are identical - should result in zero features
        W = np.ones((20, 3))
        H = np.ones((3, 15))
        
        features = compute_features(W, H)
        
        # Should be all zeros due to zero variance
        assert np.allclose(features, 0)


class TestNMFFeatureExtractor:
    """Test NMFFeatureExtractor class."""

    def test_feature_extractor_initialization(self):
        """Test NMFFeatureExtractor initialization."""
        extractor = NMFFeatureExtractor(n_components=6, use_sklearn=True)
        
        assert extractor.n_components == 6
        assert extractor.use_sklearn is True
        assert extractor.W is None
        assert extractor.H is None

    def test_feature_extractor_extract(self, mel_spectrogram):
        """Test feature extraction from mel spectrogram."""
        extractor = NMFFeatureExtractor(n_components=6, use_sklearn=True)
        
        features = extractor.extract(mel_spectrogram)
        
        assert len(features) == 6 * 8
        assert np.all(features >= 0)
        assert np.all(features <= 1)

    def test_feature_extractor_extract_stores_components(self, mel_spectrogram):
        """Test that extract stores W and H components."""
        extractor = NMFFeatureExtractor(n_components=4, use_sklearn=True)
        
        features = extractor.extract(mel_spectrogram)
        
        W, H = extractor.get_components()
        
        assert W is not None
        assert H is not None
        assert W.shape[1] == 4
        assert H.shape[0] == 4

    def test_feature_extractor_get_components_before_extract(self):
        """Test get_components returns None before extraction."""
        extractor = NMFFeatureExtractor()
        
        W, H = extractor.get_components()
        
        assert W is None
        assert H is None

    def test_feature_extractor_custom_nmf(self, mel_spectrogram):
        """Test feature extractor with custom NMF."""
        extractor = NMFFeatureExtractor(
            n_components=4,
            use_sklearn=False,
            max_iter=50,
        )
        
        features = extractor.extract(mel_spectrogram)
        
        assert len(features) == 4 * 8
        assert np.all(features >= 0)
        assert np.all(features <= 1)

    @pytest.mark.parametrize("n_components", [1, 2, 4, 6])
    def test_feature_extractor_different_components(self, mel_spectrogram, n_components):
        """Test feature extractor with different component counts."""
        extractor = NMFFeatureExtractor(n_components=n_components)
        
        features = extractor.extract(mel_spectrogram)
        
        assert len(features) == n_components * 8


class TestNMFReproducibility:
    """Test reproducibility and consistency of NMF extraction."""

    def test_sklearn_nmf_reproducible(self, mel_spectrogram):
        """Test that sklearn NMF with seed is reproducible."""
        extractor1 = NMFFeatureExtractor(n_components=4, use_sklearn=True)
        extractor2 = NMFFeatureExtractor(n_components=4, use_sklearn=True)
        
        features1 = extractor1.extract(mel_spectrogram.copy())
        features2 = extractor2.extract(mel_spectrogram.copy())
        
        # With deterministic seed, should be very close
        assert np.allclose(features1, features2, rtol=1e-5)

    def test_custom_nmf_reproducible(self, mel_spectrogram):
        """Test that custom NMF with seed is reproducible."""
        extractor1 = NMFFeatureExtractor(n_components=4, use_sklearn=False)
        extractor2 = NMFFeatureExtractor(n_components=4, use_sklearn=False)
        
        features1 = extractor1.extract(mel_spectrogram.copy())
        features2 = extractor2.extract(mel_spectrogram.copy())
        
        # Should be identical due to seed
        assert np.allclose(features1, features2)


class TestNMFEdgeCases:
    """Test edge cases for NMF feature extraction."""

    def test_nmf_single_component(self, mel_spectrogram):
        """Test NMF with single component."""
        extractor = NMFFeatureExtractor(n_components=1)
        features = extractor.extract(mel_spectrogram)
        
        assert len(features) == 1 * 8
        assert np.all(features >= 0)
        assert np.all(features <= 1)

    def test_nmf_many_components(self, mel_spectrogram):
        """Test NMF with many components."""
        n_components = 12
        extractor = NMFFeatureExtractor(n_components=n_components)
        features = extractor.extract(mel_spectrogram)
        
        assert len(features) == n_components * 8

    def test_nmf_small_spectrogram(self):
        """Test NMF on small mel spectrogram."""
        small_mel = np.random.rand(8, 4) + 0.1
        extractor = NMFFeatureExtractor(n_components=2)
        features = extractor.extract(small_mel)
        
        assert len(features) == 2 * 8

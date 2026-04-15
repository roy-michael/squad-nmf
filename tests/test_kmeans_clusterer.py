"""
Tests for K-means clustering module.

Tests unsupervised clustering, feature extraction, and cluster analysis utilities.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kmeans_clusterer import KMeansClusterer, ClusteringResult, cluster_audio_directory
from feature_extractor import extract_nmf_features
from preprocessor import AudioPreprocessor


class TestClusteringResult:
    """Test ClusteringResult dataclass."""

    def test_clustering_result_creation(self):
        """ClusteringResult should store all required fields."""
        labels = np.array([0, 1, 2, 0, 1])
        centroids = np.random.randn(3, 48)

        result = ClusteringResult(
            labels=labels,
            centroids=centroids,
            n_clusters=3,
            silhouette_score=0.5,
            davies_bouldin_score=1.2,
            inertia=100.0,
            feature_names=[f"f_{i}" for i in range(48)],
        )

        assert result.n_clusters == 3
        assert len(result.labels) == 5
        assert result.silhouette_score == 0.5
        assert len(result.feature_names) == 48


class TestKMeansClusterer:
    """Test KMeansClusterer class."""

    @pytest.fixture
    def clusterer(self):
        """Create a KMeansClusterer instance."""
        return KMeansClusterer(n_clusters=3, random_state=42)

    @pytest.fixture
    def sample_features(self):
        """Create sample features (synthetic)."""
        # Create 3 clusters with distinct means
        cluster1 = np.random.RandomState(42).randn(5, 48) + np.array([1.0] * 48)
        cluster2 = np.random.RandomState(43).randn(5, 48) + np.array([-1.0] * 48)
        cluster3 = np.random.RandomState(44).randn(5, 48)

        return np.vstack([cluster1, cluster2, cluster3])

    @pytest.fixture
    def sample_files(self):
        """Create sample file list."""
        return [f"audio_{i}.wav" for i in range(15)]

    def test_clusterer_initialization(self, clusterer):
        """Clusterer should initialize with correct parameters."""
        assert clusterer.n_clusters == 3
        assert clusterer.random_state == 42
        assert clusterer.kmeans is None

    def test_fit(self, clusterer, sample_features, sample_files):
        """fit() should cluster features and return ClusteringResult."""
        result = clusterer.fit(sample_features, sample_files)

        assert isinstance(result, ClusteringResult)
        assert len(result.labels) == 15
        assert result.n_clusters == 3
        assert result.silhouette_score > -1.0
        assert result.silhouette_score <= 1.0
        assert clusterer.kmeans is not None

    def test_fit_insufficient_samples(self, clusterer):
        """fit() should raise ValueError if samples < n_clusters."""
        features = np.random.randn(2, 48)
        with pytest.raises(ValueError):
            clusterer.fit(features)

    def test_predict(self, clusterer, sample_features):
        """predict() should assign clusters to new features."""
        clusterer.fit(sample_features)
        new_features = np.random.randn(5, 48)
        predictions = clusterer.predict(new_features)

        assert len(predictions) == 5
        assert all(0 <= label < 3 for label in predictions)

    def test_predict_before_fit(self, clusterer):
        """predict() should raise ValueError if called before fit()."""
        features = np.random.randn(5, 48)
        with pytest.raises(ValueError):
            clusterer.predict(features)

    def test_get_cluster_info(self, clusterer, sample_features, sample_files):
        """get_cluster_info() should return cluster membership details."""
        result = clusterer.fit(sample_features, sample_files)
        info = clusterer.get_cluster_info(result)

        assert len(info) == 3
        total_samples = sum(info[i]["size"] for i in range(3))
        assert total_samples == 15

        for cluster_id in range(3):
            assert "size" in info[cluster_id]
            assert "samples" in info[cluster_id]
            assert len(info[cluster_id]["samples"]) == info[cluster_id]["size"]

    def test_find_optimal_clusters(self, clusterer, sample_features):
        """find_optimal_clusters() should test different k values."""
        scores = clusterer.find_optimal_clusters(sample_features, k_range=(2, 5))

        assert len(scores) == 4
        for k in range(2, 6):
            assert k in scores
            assert -1.0 <= scores[k] <= 1.0

    def test_find_optimal_clusters_min_range(self, clusterer, sample_features):
        """find_optimal_clusters() should handle single k value."""
        scores = clusterer.find_optimal_clusters(sample_features, k_range=(3, 3))
        assert len(scores) == 1
        assert 3 in scores

    def test_cluster_consistency(self, clusterer, sample_features):
        """Clustering should be consistent with same random seed."""
        result1 = clusterer.fit(sample_features)
        labels1 = result1.labels.copy()

        clusterer2 = KMeansClusterer(n_clusters=3, random_state=42)
        result2 = clusterer2.fit(sample_features)
        labels2 = result2.labels

        # Labels may be permuted, but cluster structure should be same
        # (check by verifying inertia is same)
        assert np.isclose(result1.inertia, result2.inertia)

    def test_scaling_applied(self, clusterer, sample_features):
        """Features should be scaled before clustering."""
        clusterer.fit(sample_features)
        assert clusterer.features_raw is not None
        assert clusterer.features_scaled is not None

        # Scaled features should have mean ≈ 0, std ≈ 1
        scaled_mean = np.mean(clusterer.features_scaled, axis=0)
        scaled_std = np.std(clusterer.features_scaled, axis=0)

        assert np.allclose(scaled_mean, 0, atol=1e-10)
        assert np.allclose(scaled_std, 1.0, atol=1e-10)


class TestClusteringEdgeCases:
    """Test edge cases and robustness."""

    def test_single_feature_dimension(self):
        """Clustering should work with 1D features."""
        features = np.array([[1], [2], [10], [11], [20], [21]])
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit(features)

        assert result.n_clusters == 3
        assert len(result.labels) == 6

    def test_identical_features(self):
        """Clustering should handle identical features gracefully."""
        features = np.ones((5, 48))
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        result = clusterer.fit(features)

        assert result.n_clusters == 2
        assert len(result.labels) == 5
        # When all features are identical, K-means may put all in one cluster
        # So silhouette score will be NaN
        assert np.isnan(result.silhouette_score) or result.silhouette_score < 0

    def test_large_feature_range(self):
        """Clustering should handle large feature ranges after scaling."""
        features = np.random.randn(10, 48) * 1000  # Large scale
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit(features)

        assert result.n_clusters == 3
        assert np.isfinite(result.silhouette_score)

    def test_few_clusters_many_samples(self):
        """Clustering should handle imbalanced cluster sizes."""
        # Most samples in one cluster
        cluster1 = np.random.randn(50, 48)
        cluster2 = np.random.randn(2, 48) + 5

        features = np.vstack([cluster1, cluster2])
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        result = clusterer.fit(features)

        # Check that clustering found both clusters
        assert len(np.unique(result.labels)) == 2
        sizes = np.bincount(result.labels)
        # Larger cluster should have ~50 samples
        assert max(sizes) > 30


class TestClusteringMetrics:
    """Test clustering quality metrics."""

    def test_silhouette_score_bounds(self):
        """Silhouette score should be between -1 and 1."""
        # Create well-separated clusters
        cluster1 = np.random.randn(10, 48) + np.array([10.0] * 48)
        cluster2 = np.random.randn(10, 48) - np.array([10.0] * 48)

        features = np.vstack([cluster1, cluster2])
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        result = clusterer.fit(features)

        assert -1.0 <= result.silhouette_score <= 1.0
        # Well-separated clusters should have positive silhouette
        assert result.silhouette_score > 0

    def test_davies_bouldin_index(self):
        """Davies-Bouldin index should be non-negative."""
        features = np.random.randn(20, 48)
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit(features)

        assert result.davies_bouldin_score >= 0

    def test_inertia_decreases_with_k(self):
        """Inertia should generally decrease as k increases (with well-separated data)."""
        # Create well-separated clusters to ensure inertia decreases
        cluster1 = np.random.RandomState(42).randn(10, 48) + np.array([5.0] * 48)
        cluster2 = np.random.RandomState(43).randn(10, 48) - np.array([5.0] * 48)
        cluster3 = np.random.RandomState(44).randn(10, 48)

        features = np.vstack([cluster1, cluster2, cluster3])

        inertias = []
        for k in range(1, 6):
            clusterer = KMeansClusterer(n_clusters=k, random_state=42)
            result = clusterer.fit(features)
            inertias.append(result.inertia)

        # Check monotonic decrease (allowing small tolerance for numerical errors)
        for i in range(len(inertias) - 1):
            assert inertias[i + 1] <= inertias[i] + 1e-5, f"Inertia increased: {inertias[i]} -> {inertias[i+1]}"

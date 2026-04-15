"""
K-Means Clustering for Unsupervised Sound Discovery

Discovers natural groupings in unlabeled underwater recordings using K-means clustering
on NMF-extracted features. Enables exploratory analysis and manual labeling workflows.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from joblib import Parallel, delayed

try:
    from .feature_extractor import NMFFeatureExtractor
    from .preprocessor import AudioPreprocessor
    from .audio_loader import load_audio
except ImportError:
    # For direct imports during testing
    from feature_extractor import NMFFeatureExtractor
    from preprocessor import AudioPreprocessor
    from audio_loader import load_audio

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Result from K-means clustering analysis."""
    labels: np.ndarray  # Cluster assignment for each sample
    centroids: np.ndarray  # Cluster centers
    n_clusters: int  # Number of clusters
    silhouette_score: float  # Silhouette coefficient (-1 to 1)
    davies_bouldin_score: float  # Davies-Bouldin index (lower is better)
    inertia: float  # Sum of squared distances to nearest centroid
    feature_names: List[str]  # Names of extracted features


class KMeansClusterer:
    """
    Unsupervised clustering for underwater sound discovery.
    
    Clusters NMF-extracted features using K-means to identify natural groupings
    in unlabeled audio recordings. Useful for exploratory analysis and preparing
    data for supervised learning after manual labeling.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        """
        Initialize K-means clusterer.

        Args:
            n_clusters: Number of clusters to find
            random_state: Seed for reproducibility
            n_init: Number of KMeans initializations
            max_iter: Maximum iterations for convergence
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.scaler = StandardScaler()
        self.features_raw = None
        self.features_scaled = None
        self.audio_files = None

    def fit(
        self,
        features: np.ndarray,
        audio_files: Optional[List[str]] = None,
    ) -> ClusteringResult:
        """
        Fit K-means clustering to features.

        Args:
            features: (N, D) array of NMF features from N audio files
            audio_files: Optional list of audio file paths for reference

        Returns:
            ClusteringResult with cluster assignments and metrics
        """
        if features.shape[0] < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} samples, got {features.shape[0]}"
            )

        self.features_raw = features.copy()
        self.audio_files = audio_files or [f"audio_{i}" for i in range(len(features))]

        # Standardize features for K-means
        self.features_scaled = self.scaler.fit_transform(features)

        # Fit K-means
        logger.info(
            f"Fitting K-means with n_clusters={self.n_clusters} on {features.shape[0]} samples"
        )
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        labels = self.kmeans.fit_predict(self.features_scaled)

        # Calculate metrics (handle edge cases where silhouette cannot be computed)
        try:
            silhouette = silhouette_score(self.features_scaled, labels)
        except ValueError:
            # Happens when all samples assigned to one cluster
            silhouette = np.nan

        try:
            davies_bouldin = davies_bouldin_score(self.features_scaled, labels)
        except ValueError:
            davies_bouldin = np.nan

        logger.info(f"Silhouette Score: {silhouette:.3f}" if np.isfinite(silhouette) else "Silhouette Score: NaN (all samples in one cluster)")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.3f}" if np.isfinite(davies_bouldin) else "Davies-Bouldin Index: NaN")
        logger.info(f"Cluster distribution: {np.bincount(labels)}")

        result = ClusteringResult(
            labels=labels,
            centroids=self.kmeans.cluster_centers_,
            n_clusters=self.n_clusters,
            silhouette_score=silhouette,
            davies_bouldin_score=davies_bouldin,
            inertia=self.kmeans.inertia_,
            feature_names=[f"feature_{i}" for i in range(features.shape[1])],
        )

        return result

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new features.

        Args:
            features: (N, D) array of NMF features

        Returns:
            Cluster labels (0 to n_clusters-1)
        """
        if self.kmeans is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")

        features_scaled = self.scaler.transform(features)
        return self.kmeans.predict(features_scaled)

    def get_cluster_info(self, result: ClusteringResult) -> Dict[int, Dict]:
        """
        Get detailed information about each cluster.

        Args:
            result: ClusteringResult from fit()

        Returns:
            Dictionary mapping cluster ID to size and sample files
        """
        info = {}
        for cluster_id in range(self.n_clusters):
            mask = result.labels == cluster_id
            sample_indices = np.where(mask)[0]

            info[cluster_id] = {
                "size": int(np.sum(mask)),
                "samples": [
                    self.audio_files[i] if self.audio_files else f"sample_{i}"
                    for i in sample_indices
                ],
            }

        return info

    def find_optimal_clusters(
        self,
        features: np.ndarray,
        k_range: Tuple[int, int] = (2, 10),
    ) -> Dict[int, float]:
        """
        Test different cluster counts and return silhouette scores.

        Useful for determining optimal number of clusters.

        Args:
            features: (N, D) array of NMF features
            k_range: Tuple of (min_k, max_k) to test

        Returns:
            Dictionary mapping cluster count to silhouette score
        """
        features_scaled = self.scaler.fit_transform(features)
        scores = {}

        n_samples = features.shape[0]
        max_k = min(k_range[1], n_samples - 1)

        if max_k < k_range[0]:
            logger.warning(f"Not enough samples ({n_samples}) to test clustering. Need at least {k_range[0] + 1}.")
            return scores
            
        if max_k < k_range[1]:
            logger.info(f"Adjusted max cluster count from {k_range[1]} to {max_k} due to sample size ({n_samples})")

        logger.info(f"Testing cluster counts from {k_range[0]} to {max_k}")

        for k in range(k_range[0], max_k + 1):
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            labels = km.fit_predict(features_scaled)
            try:
                score = silhouette_score(features_scaled, labels)
                scores[k] = score
                logger.info(f"  k={k}: silhouette={score:.3f}")
            except ValueError as e:
                logger.warning(f"  k={k}: Could not calculate silhouette score - {e}")

        return scores


def _process_single_audio_file(
    wav_file: Path,
    preprocessor: AudioPreprocessor,
    feature_extractor: NMFFeatureExtractor,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Process a single audio file for feature extraction.
    
    Helper for parallel processing. Returns (features, filepath) or (None, filepath) on error.
    """
    try:
        result = load_audio(str(wav_file))
        if result is None or len(result) != 2:
            return None, str(wav_file)
        
        audio, sr = result
        if audio is None:
            return None, str(wav_file)
        
        spectrogram = preprocessor.preprocess(audio, sr)
        features = feature_extractor.extract(spectrogram)
        return features, str(wav_file)
    
    except Exception as e:
        logger.warning(f"Failed to process {wav_file.name}: {e}")
        return None, str(wav_file)


def cluster_audio_directory(
    data_dir: Path,
    n_clusters: int = 3,
    config: Optional[Dict] = None,
) -> Tuple[ClusteringResult, KMeansClusterer, List[str]]:
    """
    Convenience function to load audio directory and run clustering.

    Args:
        data_dir: Directory containing WAV files
        n_clusters: Number of clusters to find
        config: Configuration dict (uses defaults if None)

    Returns:
        Tuple of (ClusteringResult, KMeansClusterer, list of audio files)
    """
    # Default config
    if config is None:
        config = {
            "preprocessor": {
                "n_fft": 8192,
                "min_freq": 200,
                "max_freq": 12000,
                "n_mels": 128,
                "noise_gate_multiplier": 1.5,
                "hpss_margin": 3.0,
            },
            "nmf": {"n_components": 6, "use_sklearn": True, "max_iter": 2000},
        }

    # Load audio files
    preprocessor = AudioPreprocessor(**config["preprocessor"])
    feature_extractor = NMFFeatureExtractor(**config["nmf"])

    wav_files = sorted(Path(data_dir).glob("*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files in {data_dir}")

    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {data_dir}")

    # Extract features using parallel processing
    logger.info(f"Extracting features from {len(wav_files)} files (parallel processing)...")
    
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_process_single_audio_file)(wav_file, preprocessor, feature_extractor)
        for wav_file in wav_files
    )
    
    # Collect valid results
    features_list = []
    valid_files = []
    for features, filepath in results:
        if features is not None:
            features_list.append(features)
            valid_files.append(filepath)
            logger.debug(f"Extracted features from {Path(filepath).name}")
    
    features = np.array(features_list)
    logger.info(f"Extracted {features.shape[1]}-dim features from {len(valid_files)} files")

    # Cluster
    clusterer = KMeansClusterer(n_clusters=n_clusters)
    result = clusterer.fit(features, valid_files)

    return result, clusterer, valid_files

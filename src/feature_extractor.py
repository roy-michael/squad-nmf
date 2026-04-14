"""
NMF-based feature extraction module for underwater vehicle sound classification.

This module converts preprocessed mel spectrograms into machine-learning-ready
feature vectors using Non-negative Matrix Factorization (NMF).
"""

from typing import Tuple
import numpy as np
from scipy.stats import skew, kurtosis

try:
    from sklearn.decomposition import NMF as SklearnNMF
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def nmf_multiplicative(
    V: np.ndarray,
    k: int,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Non-negative Matrix Factorization using multiplicative update rules.

    Factors matrix V into W and H such that V ≈ W * H using iterative
    multiplicative updates with early convergence detection.

    Args:
        V: Input matrix (n_features × m_samples)
        k: Number of components (rank of factorization)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (Frobenius norm difference)

    Returns:
        Tuple of:
            - W: Basis functions (n_features × k)
            - H: Activations (k × m_samples)
            - errors: List of reconstruction errors per iteration
    """
    n, m = V.shape

    np.random.seed(42)
    W = np.random.rand(n, k)
    H = np.random.rand(k, m)

    errors = []

    for i in range(max_iter):
        # Update H (activations)
        numerator_H = W.T @ V
        denominator_H = W.T @ W @ H + 1e-9
        H = H * (numerator_H / denominator_H)

        # Update W (basis functions)
        numerator_W = V @ H.T
        denominator_W = W @ H @ H.T + 1e-9
        W = W * (numerator_W / denominator_W)

        # Compute reconstruction error
        error = np.linalg.norm(V - W @ H, "fro")
        errors.append(error)

        # Early stopping
        if i > 0 and abs(errors[-2] - error) < tol:
            break

    return W, H, errors


def extract_nmf_features(
    S_mel: np.ndarray,
    n_components: int = 6,
    use_sklearn: bool = True,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Decompose mel spectrogram using NMF into basis functions and activations.

    Args:
        S_mel: Denoised mel spectrogram (mel_bins × time_steps)
        n_components: Number of components for NMF decomposition
        use_sklearn: Use sklearn NMF (True) or custom multiplicative updates (False)
        max_iter: Maximum iterations for NMF algorithm

    Returns:
        Tuple of:
            - W: Basis functions (mel_bins × n_components)
            - H: Activations (n_components × time_steps)
            - errors: Convergence errors for diagnostics
    """
    if use_sklearn:
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for use_sklearn=True")
        model = SklearnNMF(
            n_components=n_components,
            init="nndsvda",
            solver="cd",
            max_iter=max_iter,
            random_state=42,
        )
        W = model.fit_transform(S_mel)
        H = model.components_
        errors = []
    else:
        W, H, errors = nmf_multiplicative(S_mel, n_components, max_iter=max_iter)

    return W, H, errors


def compute_features(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Extract statistical moments from NMF basis functions and activations.

    Computes mean, std, skewness, and kurtosis for each component in both
    W (basis functions) and H (activations), then concatenates and normalizes
    to [0, 1] range.

    Args:
        W: Basis functions (freq_bins × n_components)
        H: Activations (n_components × time_steps)

    Returns:
        Feature vector (n_components × 8,) with moments from W and H:
            - Elements 0-3: mean, std, skew, kurtosis from W
            - Elements 4-7: mean, std, skew, kurtosis from H
    """
    n_components = W.shape[1]
    features = []

    # Extract statistical moments from basis functions (W)
    for i in range(n_components):
        w_col = W[:, i]
        moments_w = [
            np.mean(w_col),
            np.std(w_col),
            skew(w_col),
            kurtosis(w_col),
        ]
        features.extend(moments_w)

    # Extract statistical moments from activations (H)
    for i in range(n_components):
        h_row = H[i, :]
        moments_h = [
            np.mean(h_row),
            np.std(h_row),
            skew(h_row),
            kurtosis(h_row),
        ]
        features.extend(moments_h)

    # Convert to numpy array
    feature_vector = np.array(features, dtype=np.float32)

    # Normalize to [0, 1] using min-max scaling
    min_val = np.min(feature_vector)
    max_val = np.max(feature_vector)

    if max_val > min_val:
        feature_vector = (feature_vector - min_val) / (max_val - min_val)
    else:
        # Handle case where all features are identical
        feature_vector = np.zeros_like(feature_vector)

    return feature_vector


class NMFFeatureExtractor:
    """
    Integration class for full NMF-based feature extraction pipeline.

    Handles conversion of mel spectrograms to ML-ready feature vectors
    through NMF decomposition and statistical feature extraction.
    """

    def __init__(
        self,
        n_components: int = 6,
        use_sklearn: bool = True,
        max_iter: int = 500,
    ):
        """
        Initialize NMF feature extractor.

        Args:
            n_components: Number of NMF components
            use_sklearn: Use sklearn NMF (True) or custom implementation (False)
            max_iter: Maximum iterations for NMF algorithm
        """
        self.n_components = n_components
        self.use_sklearn = use_sklearn
        self.max_iter = max_iter
        self.W = None
        self.H = None
        self.errors = None

    def extract(self, S_mel: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from mel spectrogram.

        Performs full extraction pipeline: NMF decomposition followed by
        statistical feature computation.

        Args:
            S_mel: Denoised mel spectrogram (mel_bins × time_steps)

        Returns:
            Feature vector (n_components × 8,) normalized to [0, 1]
        """
        # Decompose mel spectrogram using NMF
        self.W, self.H, self.errors = extract_nmf_features(
            S_mel,
            n_components=self.n_components,
            use_sklearn=self.use_sklearn,
            max_iter=self.max_iter,
        )

        # Compute features from W and H
        feature_vector = compute_features(self.W, self.H)

        return feature_vector

    def get_components(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve NMF components from last extraction.

        Useful for visualization and diagnostics.

        Returns:
            Tuple of (W, H) from most recent extraction, or (None, None)
            if no extraction has been performed yet.
        """
        return self.W, self.H

"""
Audio preprocessing module with denoising and feature extraction pipeline.
"""
import warnings
from typing import Dict, Any
import numpy as np
import librosa

try:
    from .welch import Analysis
except ImportError:
    from welch import Analysis


class AudioPreprocessor(Analysis):
    """
    Audio preprocessing pipeline with denoising and Mel-scale feature extraction.

    Applies STFT, spectral gating, harmonic-percussive source separation (HPSS),
    and Mel-scale transformation for audio classification.

    Parameters
    ----------
    min_freq : float, optional
        Minimum frequency in Hz (default: 200).
    max_freq : float, optional
        Maximum frequency in Hz (default: 12000).
    n_fft : int, optional
        FFT size for STFT (default: 8192).
    n_mels : int, optional
        Number of Mel frequency bands (default: 128).
    noise_gate_multiplier : float, optional
        Multiplier for noise profile subtraction (default: 1.5).
    hpss_margin : float, optional
        Margin parameter for HPSS decomposition (default: 3.0).
    """

    def __init__(
        self,
        min_freq: float = 200,
        max_freq: float = 12000,
        n_fft: int = 8192,
        n_mels: int = 128,
        noise_gate_multiplier: float = 1.5,
        hpss_margin: float = 3.0,
    ):
        """Initialize the audio preprocessor."""
        super().__init__(min_freq, max_freq)
        self.n_fft = n_fft
        self.hop_length = n_fft // 16
        self.n_mels = n_mels
        self.noise_gate_multiplier = noise_gate_multiplier
        self.hpss_margin = hpss_margin

        # These will be set during preprocessing
        self.actual_max_freq = None
        self.actual_n_mels = None

    def _validate_and_adjust_params(self, sr: int) -> None:
        """
        Validate and adjust preprocessing parameters based on sample rate.

        Parameters
        ----------
        sr : int
            Sample rate in Hz.

        Raises
        ------
        ValueError
            If sample rate is invalid or frequency range is invalid.
        """
        if sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")

        nyquist_freq = sr / 2.0
        if self.max_freq > nyquist_freq:
            self.actual_max_freq = nyquist_freq
            warnings.warn(
                f"max_freq ({self.max_freq} Hz) exceeds Nyquist frequency ({nyquist_freq} Hz). "
                f"Clamping to {nyquist_freq} Hz.",
                UserWarning,
            )
        else:
            self.actual_max_freq = self.max_freq

        if self.min_freq >= self.actual_max_freq:
            raise ValueError(
                f"min_freq ({self.min_freq} Hz) must be less than max_freq "
                f"({self.actual_max_freq} Hz)"
            )

    def _adjust_n_mels(self, sr: int) -> None:
        """
        Dynamically adjust n_mels if frequency resolution is too coarse.

        Ensures that Mel filterbank spacing is not finer than the STFT frequency
        resolution to avoid empty or near-empty filters.

        Parameters
        ----------
        sr : int
            Sample rate in Hz.
        """
        freq_resolution = sr / self.n_fft

        # Test if current n_mels is feasible
        mel_freqs = librosa.mel_frequencies(
            n_mels=self.n_mels + 2, fmin=self.min_freq, fmax=self.actual_max_freq
        )
        mel_freqs_diff = np.diff(mel_freqs)

        # If any mel band spacing is finer than STFT resolution, reduce n_mels
        if np.any(mel_freqs_diff < freq_resolution):
            for n in range(self.n_mels, 1, -1):
                mel_freqs = librosa.mel_frequencies(
                    n_mels=n + 2, fmin=self.min_freq, fmax=self.actual_max_freq
                )
                mel_freqs_diff = np.diff(mel_freqs)
                if np.all(mel_freqs_diff >= freq_resolution):
                    self.actual_n_mels = n
                    warnings.warn(
                        f"n_mels={self.n_mels} is too high for the given frequency range "
                        f"and n_fft={self.n_fft}. Reducing n_mels to {self.actual_n_mels}.",
                        UserWarning,
                    )
                    return

            # Fallback: use n_mels=1 if we can't find a suitable value
            self.actual_n_mels = 1
            warnings.warn(
                f"Could not find suitable n_mels. Using n_mels=1.",
                UserWarning,
            )
        else:
            self.actual_n_mels = self.n_mels

    def _stft(self, y: np.ndarray) -> np.ndarray:
        """
        Compute magnitude spectrogram using Short-Time Fourier Transform.

        Parameters
        ----------
        y : np.ndarray
            Audio signal (mono).

        Returns
        -------
        S : np.ndarray
            Magnitude spectrogram (n_fft // 2 + 1, n_frames).
        """
        return np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))

    def _spectral_gating(self, S: np.ndarray) -> np.ndarray:
        """
        Apply spectral gating (median-based noise subtraction).

        Computes a median noise profile and subtracts it from the spectrogram
        to suppress background noise.

        Parameters
        ----------
        S : np.ndarray
            Magnitude spectrogram.

        Returns
        -------
        S_gated : np.ndarray
            Noise-gated spectrogram (non-negative).
        """
        noise_profile = np.median(S, axis=1, keepdims=True)
        S_gated = S - (noise_profile * self.noise_gate_multiplier)
        S_gated = np.maximum(0, S_gated)
        return S_gated

    def _hpss_decomposition(self, S: np.ndarray) -> np.ndarray:
        """
        Apply Harmonic-Percussive Source Separation.

        Decomposes the spectrogram into harmonic and percussive components,
        returning only the harmonic component for further processing.

        Parameters
        ----------
        S : np.ndarray
            Magnitude spectrogram.

        Returns
        -------
        S_harmonic : np.ndarray
            Harmonic component of the spectrogram.
        """
        S_harmonic, _ = librosa.decompose.hpss(S, margin=self.hpss_margin)
        return S_harmonic

    def _mel_transform(self, S: np.ndarray, sr: int) -> np.ndarray:
        """
        Convert spectrogram to Mel scale.

        Parameters
        ----------
        S : np.ndarray
            Linear-scale spectrogram.
        sr : int
            Sample rate in Hz.

        Returns
        -------
        S_mel : np.ndarray
            Mel-scale spectrogram (n_mels, n_frames).
        """
        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=self.n_fft,
            n_mels=self.actual_n_mels,
            fmin=self.min_freq,
            fmax=self.actual_max_freq,
        )
        return np.dot(mel_basis, S)

    def preprocess(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply full preprocessing pipeline to audio signal.

        Applies STFT, spectral gating, HPSS decomposition, and Mel-scale
        transformation to produce a denoised Mel spectrogram ready for
        feature extraction or classification.

        Parameters
        ----------
        y : np.ndarray
            Audio signal (mono).
        sr : int
            Sample rate in Hz.

        Returns
        -------
        S_mel : np.ndarray
            Denoised Mel-scale spectrogram (actual_n_mels, n_frames).

        Raises
        ------
        ValueError
            If audio or sample rate are invalid.
        """
        # Validate inputs
        if y.ndim != 1:
            raise ValueError(f"Audio must be 1D, got shape {y.shape}")

        if len(y) == 0:
            raise ValueError("Audio signal is empty")

        # Validate and adjust parameters
        self._validate_and_adjust_params(sr)
        self._adjust_n_mels(sr)

        # Apply preprocessing pipeline
        S = self._stft(y)
        S_gated = self._spectral_gating(S)
        S_harmonic = self._hpss_decomposition(S_gated)
        S_mel = self._mel_transform(S_harmonic, sr)

        return S_mel

    def get_info(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration and applied parameters.

        Returns
        -------
        info : dict
            Dictionary containing:
            - 'n_fft': FFT size
            - 'hop_length': Hop length
            - 'n_mels': Requested number of Mel bands
            - 'actual_n_mels': Actual number of Mel bands (after adjustment)
            - 'min_freq': Minimum frequency (Hz)
            - 'max_freq': Maximum frequency (Hz)
            - 'actual_max_freq': Actual max frequency (after Nyquist check)
            - 'noise_gate_multiplier': Noise gating multiplier
            - 'hpss_margin': HPSS margin parameter
        """
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "actual_n_mels": self.actual_n_mels,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "actual_max_freq": self.actual_max_freq,
            "noise_gate_multiplier": self.noise_gate_multiplier,
            "hpss_margin": self.hpss_margin,
        }

    def _analyse(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Internal analysis method (required by Analysis base class).

        Parameters
        ----------
        y : np.ndarray
            Audio signal (mono).
        sr : int
            Sample rate in Hz.

        Returns
        -------
        S_mel : np.ndarray
            Denoised Mel-scale spectrogram.
        """
        return self.preprocess(y, sr)

    def plot(self, y: np.ndarray, sr: int, file_id: str) -> None:
        """
        Plot preprocessing results (placeholder for future visualization).

        Parameters
        ----------
        y : np.ndarray
            Audio signal (mono).
        sr : int
            Sample rate in Hz.
        file_id : str
            Identifier for the audio file (for plotting titles).

        Notes
        -----
        This is a placeholder implementation of the abstract plot method.
        Full visualization can be added as needed.
        """
        pass

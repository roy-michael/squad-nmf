"""
Base analysis class for spectral analysis of audio signals.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from abc import ABC, abstractmethod


class Analysis(ABC):
    """Abstract base class for spectral analysis methods."""

    def __init__(self, min_freq, max_freq):
        """
        Initialize the Analysis base class.

        Parameters
        ----------
        min_freq : float
            Minimum frequency in Hz for the analysis window.
        max_freq : float
            Maximum frequency in Hz for the analysis window.
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.step = 200

    @abstractmethod
    def _analyse(self, y, sr):
        """
        Abstract method to perform spectral analysis.

        Parameters
        ----------
        y : np.ndarray
            Audio signal (mono).
        sr : int
            Sample rate in Hz.

        Returns
        -------
        Various (implementation-dependent)
            Analysis results.
        """
        pass

    @abstractmethod
    def plot(self, y, sr, file_id):
        """
        Abstract method to plot analysis results.

        Parameters
        ----------
        y : np.ndarray
            Audio signal (mono).
        sr : int
            Sample rate in Hz.
        file_id : str
            Identifier for the audio file (for plotting titles).
        """
        pass


class Welch(Analysis):
    """Welch's method for power spectral density estimation."""

    def __init__(self, min_freq=200, max_freq=12000, n_fft=8192):
        """
        Initialize Welch's periodogram analyzer.

        Parameters
        ----------
        min_freq : float, optional
            Minimum frequency in Hz (default: 200).
        max_freq : float, optional
            Maximum frequency in Hz (default: 12000).
        n_fft : int, optional
            FFT size (default: 8192).
        """
        super().__init__(min_freq, max_freq)
        self.n_fft = n_fft

    def _analyse(self, y, sr):
        """
        Compute Welch's periodogram.

        Parameters
        ----------
        y : np.ndarray
            Audio signal (mono).
        sr : int
            Sample rate in Hz.

        Returns
        -------
        f : np.ndarray
            Frequency values.
        Pxx_den : np.ndarray
            Power spectral density.
        """
        return signal.welch(y, sr, nperseg=self.n_fft)

    def plot(self, y, sr, file_id):
        """
        Plot Welch's periodogram on logarithmic frequency scale.

        Parameters
        ----------
        y : np.ndarray
            Audio signal (mono).
        sr : int
            Sample rate in Hz.
        file_id : str
            Identifier for the audio file (for plotting titles).
        """
        self.f_welch, self.Pxx_den = self._analyse(y, sr)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6), num="Welch Periodogram")
        fig.suptitle(f'Welch Periodogram ({file_id})', fontsize=16)

        # Plot: Logarithmic Frequency Scale
        ax.loglog(self.f_welch, self.Pxx_den)
        ax.set_title('Logarithmic Frequency Scale')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD [V**2/Hz]')
        ax.set_xlim(max(1, self.min_freq), self.max_freq)  # Avoid log(0)
        ax.grid(True, which="both", ls="-", alpha=0.8)

        plt.tight_layout()

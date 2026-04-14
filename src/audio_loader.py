"""
Audio loading and validation module for WAV file I/O.
"""
import os
from typing import Tuple
import numpy as np
import librosa


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a WAV audio file and validate it.

    Loads audio using librosa at its original sample rate. Handles both mono
    and stereo audio by converting stereo to mono (averaging channels).

    Parameters
    ----------
    file_path : str
        Path to the WAV file.

    Returns
    -------
    audio_data : np.ndarray
        Audio signal as a 1D numpy array (mono).
    sample_rate : int
        Sample rate in Hz.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be read or is corrupted.

    Examples
    --------
    >>> audio, sr = load_audio('path/to/file.wav')
    >>> print(f"Loaded {len(audio)} samples at {sr} Hz")
    """
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        # Load audio with original sample rate (sr=None)
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)

        # Handle stereo: convert to mono by averaging channels
        if audio_data.ndim == 2:
            # Stereo case: average across channels
            audio_data = np.mean(audio_data, axis=0)

        # Ensure output is 1D
        if audio_data.ndim != 1:
            raise ValueError(f"Unexpected audio dimensions: {audio_data.ndim}D")

        # Validate data
        if len(audio_data) == 0:
            raise ValueError("Audio file is empty")

        if not np.isfinite(audio_data).all():
            raise ValueError("Audio contains non-finite values (NaN or Inf)")

        return audio_data, sample_rate

    except librosa.LibrosaError as e:
        raise ValueError(f"Failed to read audio file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading audio: {str(e)}")

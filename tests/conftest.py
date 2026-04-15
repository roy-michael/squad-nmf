"""
Pytest configuration and shared fixtures for underwater vehicle sound classifier tests.

Provides synthetic test data and common utilities for all test modules.
"""
import os
import sys
import tempfile
import pytest
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def synthetic_audio():
    """
    Create synthetic audio signal for testing.
    
    Returns
    -------
    audio : np.ndarray
        1D synthetic audio array (mono, 2 seconds, 16 kHz)
    sr : int
        Sample rate (16000 Hz)
    """
    np.random.seed(42)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create signal with multiple frequency components
    freq1, freq2 = 440, 880
    audio = (
        0.3 * np.sin(2 * np.pi * freq1 * t) +
        0.2 * np.sin(2 * np.pi * freq2 * t) +
        0.1 * np.random.randn(len(t))
    )
    
    # Normalize to [-1, 1]
    audio = audio / (np.abs(audio).max() + 1e-8)
    
    return audio.astype(np.float32), sr


@pytest.fixture
def synthetic_wav_file(synthetic_audio, tmp_path):
    """
    Create a temporary synthetic WAV file.
    
    Parameters
    ----------
    synthetic_audio : tuple
        Fixture providing (audio, sr)
    tmp_path : Path
        pytest temporary directory
        
    Returns
    -------
    filepath : str
        Path to the created WAV file
    """
    audio, sr = synthetic_audio
    filepath = str(tmp_path / "test_audio.wav")
    sf.write(filepath, audio, sr)
    return filepath


@pytest.fixture
def stereo_audio():
    """
    Create stereo synthetic audio signal.
    
    Returns
    -------
    audio : np.ndarray
        2D stereo audio array (2, n_samples)
    sr : int
        Sample rate (16000 Hz)
    """
    np.random.seed(42)
    sr = 16000
    duration = 1.0
    n_samples = int(sr * duration)
    
    t = np.linspace(0, duration, n_samples)
    
    # Left channel
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Right channel (slightly different)
    right = 0.3 * np.sin(2 * np.pi * 550 * t)
    
    stereo = np.vstack([left, right]).astype(np.float32)
    
    return stereo, sr


@pytest.fixture
def stereo_wav_file(stereo_audio, tmp_path):
    """
    Create a temporary stereo WAV file.
    
    Parameters
    ----------
    stereo_audio : tuple
        Fixture providing stereo (audio, sr)
    tmp_path : Path
        pytest temporary directory
        
    Returns
    -------
    filepath : str
        Path to the created stereo WAV file
    """
    audio, sr = stereo_audio
    filepath = str(tmp_path / "test_stereo.wav")
    sf.write(filepath, audio.T, sr)  # sf.write expects (n_samples, n_channels)
    return filepath


@pytest.fixture
def short_audio():
    """
    Create very short audio signal for edge case testing.
    
    Returns
    -------
    audio : np.ndarray
        Short 1D synthetic audio array (0.2 seconds, 16 kHz)
    sr : int
        Sample rate (16000 Hz)
    """
    np.random.seed(42)
    sr = 16000
    duration = 0.2  # Very short
    audio = np.random.randn(int(sr * duration)).astype(np.float32)
    return audio / (np.abs(audio).max() + 1e-8), sr


@pytest.fixture
def short_wav_file(short_audio, tmp_path):
    """
    Create a temporary short WAV file.
    
    Parameters
    ----------
    short_audio : tuple
        Fixture providing (audio, sr)
    tmp_path : Path
        pytest temporary directory
        
    Returns
    -------
    filepath : str
        Path to the created short WAV file
    """
    audio, sr = short_audio
    filepath = str(tmp_path / "test_short.wav")
    sf.write(filepath, audio, sr)
    return filepath


@pytest.fixture
def silent_audio():
    """
    Create silent (near-zero) audio for edge case testing.
    
    Returns
    -------
    audio : np.ndarray
        1D silent audio array (1 second, 16 kHz)
    sr : int
        Sample rate (16000 Hz)
    """
    sr = 16000
    duration = 1.0
    # Create near-zero audio (very small noise floor)
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 1e-10
    return audio, sr


@pytest.fixture
def silent_wav_file(silent_audio, tmp_path):
    """
    Create a temporary silent WAV file.
    
    Parameters
    ----------
    silent_audio : tuple
        Fixture providing (audio, sr)
    tmp_path : Path
        pytest temporary directory
        
    Returns
    -------
    filepath : str
        Path to the created silent WAV file
    """
    audio, sr = silent_audio
    filepath = str(tmp_path / "test_silent.wav")
    sf.write(filepath, audio, sr)
    return filepath


@pytest.fixture
def mel_spectrogram(synthetic_audio):
    """
    Create a synthetic mel spectrogram for testing.
    
    Returns
    -------
    mel_spec : np.ndarray
        Mel spectrogram (n_mels, n_frames)
    """
    audio, sr = synthetic_audio
    n_fft = 2048
    hop_length = 512
    
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(np.abs(S) ** 2, ref=np.max)
    
    mel_spec = librosa.feature.melspectrogram(
        S=librosa.db_to_power(S_db),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
    )
    
    return mel_spec.astype(np.float32)


@pytest.fixture
def feature_vector():
    """
    Create a synthetic feature vector for testing.
    
    Returns
    -------
    features : np.ndarray
        Feature vector (n_components * 8,) normalized to [0, 1]
    """
    np.random.seed(42)
    n_components = 6
    n_features = n_components * 8
    
    # Create random features in [0, 1]
    features = np.random.uniform(0, 1, n_features).astype(np.float32)
    
    return features


@pytest.fixture
def multiple_feature_vectors():
    """
    Create multiple synthetic feature vectors for classification testing.
    
    Returns
    -------
    features : np.ndarray
        Feature matrix (n_samples, n_components * 8)
    labels : np.ndarray
        Class labels (n_samples,)
    """
    np.random.seed(42)
    n_samples = 30
    n_components = 6
    n_features = n_components * 8
    
    # Create features
    features = np.random.uniform(0, 1, (n_samples, n_features)).astype(np.float32)
    
    # Create balanced labels (3 classes)
    labels = np.repeat([0, 1, 2], n_samples // 3)
    
    return features, labels


@pytest.fixture
def invalid_audio_file(tmp_path):
    """
    Create an invalid audio file (not actually audio).
    
    Parameters
    ----------
    tmp_path : Path
        pytest temporary directory
        
    Returns
    -------
    filepath : str
        Path to the invalid file
    """
    filepath = str(tmp_path / "invalid.txt")
    with open(filepath, "w") as f:
        f.write("This is not an audio file")
    return filepath


@pytest.fixture(autouse=True)
def add_src_to_path(monkeypatch):
    """
    Add src directory to Python path for all tests.
    
    This allows imports like `from audio_loader import load_audio`.
    """
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    monkeypatch.setenv("PYTHONPATH", src_path + os.pathsep + os.environ.get("PYTHONPATH", ""))

"""
Test suite for audio_loader.py module.

Tests audio loading, validation, and format handling.
"""
import pytest
import os
import numpy as np
from audio_loader import load_audio


class TestAudioLoaderBasics:
    """Test basic audio loading functionality."""

    def test_load_valid_wav(self, synthetic_wav_file):
        """Test loading a valid WAV file."""
        audio, sr = load_audio(synthetic_wav_file)
        
        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, (int, np.integer))
        assert sr > 0
        assert len(audio) > 0
        assert audio.ndim == 1

    def test_load_mono_audio(self, synthetic_wav_file):
        """Test loading mono audio and verify 1D output."""
        audio, sr = load_audio(synthetic_wav_file)
        
        assert audio.ndim == 1, "Audio should be 1D (mono)"
        assert len(audio) > 0

    def test_load_stereo_to_mono(self, stereo_wav_file):
        """Test loading stereo audio converts to mono."""
        audio, sr = load_audio(stereo_wav_file)
        
        # Verify output is mono (1D)
        assert audio.ndim == 1, "Stereo should be converted to mono"
        assert len(audio) > 0
        assert sr > 0

    def test_sample_rate_preserved(self, synthetic_wav_file):
        """Test that sample rate is preserved during loading."""
        audio, sr = load_audio(synthetic_wav_file)
        
        # Standard sample rates
        assert sr in [8000, 16000, 22050, 44100, 48000, 96000]

    def test_audio_dimensions(self, synthetic_wav_file):
        """Test that returned audio is 1D array."""
        audio, sr = load_audio(synthetic_wav_file)
        
        assert audio.ndim == 1
        assert isinstance(audio, np.ndarray)


class TestAudioLoaderErrorHandling:
    """Test error handling in audio loader."""

    def test_invalid_file_path(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/to/audio.wav")

    def test_invalid_file_format(self, invalid_audio_file):
        """Test ValueError for invalid audio file format."""
        with pytest.raises(ValueError):
            load_audio(invalid_audio_file)

    def test_empty_file_path(self):
        """Test error with empty file path."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_audio("")

    def test_directory_instead_of_file(self, tmp_path):
        """Test error when path points to directory."""
        with pytest.raises(ValueError):
            load_audio(str(tmp_path))


class TestAudioLoaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_load_short_audio(self, short_wav_file):
        """Test loading very short audio file."""
        audio, sr = load_audio(short_wav_file)
        
        assert audio.ndim == 1
        assert len(audio) > 0
        assert sr > 0

    def test_load_silent_audio(self, silent_wav_file):
        """Test loading nearly silent audio."""
        audio, sr = load_audio(silent_wav_file)
        
        assert audio.ndim == 1
        assert len(audio) > 0
        # Verify it's actually very quiet
        assert np.abs(audio).max() < 0.001

    def test_audio_is_finite(self, synthetic_wav_file):
        """Test that loaded audio contains no NaN or Inf values."""
        audio, sr = load_audio(synthetic_wav_file)
        
        assert np.isfinite(audio).all()

    def test_audio_amplitude_range(self, synthetic_wav_file):
        """Test that audio amplitude is within reasonable range."""
        audio, sr = load_audio(synthetic_wav_file)
        
        # Audio should typically be in [-1, 1] or close to it
        assert np.abs(audio).max() <= 1.1


class TestAudioLoaderDataTypes:
    """Test data types and format consistency."""

    def test_audio_is_float_array(self, synthetic_wav_file):
        """Test that audio is returned as float array."""
        audio, sr = load_audio(synthetic_wav_file)
        
        assert audio.dtype in [np.float32, np.float64]

    def test_sample_rate_is_integer(self, synthetic_wav_file):
        """Test that sample rate is integer."""
        audio, sr = load_audio(synthetic_wav_file)
        
        assert isinstance(sr, (int, np.integer))
        assert sr > 0

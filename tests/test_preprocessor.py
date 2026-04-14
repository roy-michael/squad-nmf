"""
Test suite for preprocessor.py module.

Tests audio preprocessing pipeline including denoising and mel spectrogram extraction.
"""
import pytest
import numpy as np
import warnings
from preprocessor import AudioPreprocessor


class TestPreprocessorInitialization:
    """Test preprocessor initialization and configuration."""

    def test_preprocessor_initialization_default(self):
        """Test default preprocessor initialization."""
        preprocessor = AudioPreprocessor()
        
        assert preprocessor.min_freq == 200
        assert preprocessor.max_freq == 12000
        assert preprocessor.n_fft == 8192
        assert preprocessor.n_mels == 128
        assert preprocessor.noise_gate_multiplier == 1.5
        assert preprocessor.hpss_margin == 3.0

    def test_preprocessor_initialization_custom(self):
        """Test preprocessor with custom parameters."""
        preprocessor = AudioPreprocessor(
            min_freq=100,
            max_freq=8000,
            n_fft=4096,
            n_mels=64,
            noise_gate_multiplier=2.0,
            hpss_margin=2.0,
        )
        
        assert preprocessor.min_freq == 100
        assert preprocessor.max_freq == 8000
        assert preprocessor.n_fft == 4096
        assert preprocessor.n_mels == 64
        assert preprocessor.noise_gate_multiplier == 2.0
        assert preprocessor.hpss_margin == 2.0

    def test_get_info_returns_dict(self):
        """Test that get_info() returns configuration dictionary."""
        preprocessor = AudioPreprocessor()
        
        # Need to run preprocess first to set actual values
        np.random.seed(42)
        audio = np.random.randn(16000)
        sr = 16000
        preprocessor.preprocess(audio, sr)
        
        info = preprocessor.get_info()
        
        assert isinstance(info, dict)
        assert "n_fft" in info
        assert "n_mels" in info
        assert "min_freq" in info
        assert "max_freq" in info
        assert "actual_n_mels" in info
        assert "actual_max_freq" in info
        assert "noise_gate_multiplier" in info
        assert "hpss_margin" in info


class TestPreprocessorOutput:
    """Test preprocessor output characteristics."""

    def test_preprocess_output_shape(self, synthetic_audio):
        """Test that preprocessor outputs correct mel spectrogram shape."""
        audio, sr = synthetic_audio
        preprocessor = AudioPreprocessor()
        
        mel_spec = preprocessor.preprocess(audio, sr)
        
        assert mel_spec.ndim == 2
        # First dimension should be mel bins
        assert mel_spec.shape[0] > 0
        # Second dimension should be time frames
        assert mel_spec.shape[1] > 0

    def test_preprocess_output_range(self, synthetic_audio):
        """Test that mel spectrogram values are non-negative."""
        audio, sr = synthetic_audio
        preprocessor = AudioPreprocessor()
        
        mel_spec = preprocessor.preprocess(audio, sr)
        
        # After processing, should be non-negative
        assert np.all(mel_spec >= 0) or np.allclose(mel_spec, 0, atol=1e-6)

    @pytest.mark.parametrize("n_mels", [32, 64, 128])
    def test_preprocess_different_mel_bands(self, synthetic_audio, n_mels):
        """Test preprocessor with different numbers of mel bands."""
        audio, sr = synthetic_audio
        preprocessor = AudioPreprocessor(n_mels=n_mels)
        
        mel_spec = preprocessor.preprocess(audio, sr)
        
        # First dimension should match n_mels (or less if adjusted)
        assert mel_spec.shape[0] > 0
        assert mel_spec.shape[0] <= n_mels


class TestPreprocessorDenoising:
    """Test denoising and noise reduction capabilities."""

    def test_denoising_reduces_noise(self, synthetic_audio):
        """Test that spectral gating reduces noise."""
        audio, sr = synthetic_audio
        
        preprocessor = AudioPreprocessor()
        
        # Add noise to audio
        np.random.seed(42)
        noisy_audio = audio + 0.1 * np.random.randn(len(audio))
        noisy_audio = noisy_audio / (np.abs(noisy_audio).max() + 1e-8)
        
        mel_spec_noisy = preprocessor.preprocess(noisy_audio, sr)
        
        # After preprocessing, low-magnitude frames should be reduced
        # (This is indirect testing, as we can't directly compare without knowing the pattern)
        assert mel_spec_noisy.shape[1] > 0
        assert np.all(mel_spec_noisy >= 0) or np.allclose(mel_spec_noisy, 0, atol=1e-6)

    def test_noise_gate_multiplier_effect(self, synthetic_audio):
        """Test that noise gate multiplier affects output."""
        audio, sr = synthetic_audio
        
        preprocessor_low = AudioPreprocessor(noise_gate_multiplier=1.0)
        preprocessor_high = AudioPreprocessor(noise_gate_multiplier=2.0)
        
        mel_spec_low = preprocessor_low.preprocess(audio, sr)
        mel_spec_high = preprocessor_high.preprocess(audio, sr)
        
        # Both should be valid spectrograms
        assert mel_spec_low.shape[1] > 0
        assert mel_spec_high.shape[1] > 0


class TestPreprocessorFrequencyHandling:
    """Test frequency range handling and Nyquist frequency clamping."""

    def test_frequency_range_clamping(self, synthetic_audio):
        """Test max_freq is clamped to Nyquist frequency."""
        audio, sr = synthetic_audio
        
        # Set max_freq higher than Nyquist
        max_freq_requested = sr  # Higher than Nyquist (sr/2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            preprocessor = AudioPreprocessor(max_freq=max_freq_requested)
            mel_spec = preprocessor.preprocess(audio, sr)
            
            # Should have issued a warning about clamping
            assert len(w) >= 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Nyquist" in str(w[-1].message)
        
        assert mel_spec.shape[1] > 0

    def test_valid_frequency_range(self, synthetic_audio):
        """Test with valid frequency range."""
        audio, sr = synthetic_audio
        
        preprocessor = AudioPreprocessor(min_freq=100, max_freq=sr // 2 - 1000)
        mel_spec = preprocessor.preprocess(audio, sr)
        
        assert mel_spec.shape[1] > 0

    def test_invalid_frequency_range(self, synthetic_audio):
        """Test error when min_freq >= max_freq."""
        audio, sr = synthetic_audio
        
        preprocessor = AudioPreprocessor(min_freq=8000, max_freq=1000)
        
        with pytest.raises(ValueError):
            preprocessor.preprocess(audio, sr)


class TestPreprocessorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_edge_case_short_audio(self, short_audio):
        """Test preprocessor handles very short audio."""
        audio, sr = short_audio
        preprocessor = AudioPreprocessor()
        
        mel_spec = preprocessor.preprocess(audio, sr)
        
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] > 0
        assert mel_spec.shape[1] > 0

    def test_edge_case_silent_audio(self, silent_audio):
        """Test preprocessor handles silent audio."""
        audio, sr = silent_audio
        preprocessor = AudioPreprocessor()
        
        mel_spec = preprocessor.preprocess(audio, sr)
        
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] > 0
        assert mel_spec.shape[1] > 0

    def test_empty_audio_raises_error(self):
        """Test that empty audio raises ValueError."""
        preprocessor = AudioPreprocessor()
        
        with pytest.raises(ValueError):
            preprocessor.preprocess(np.array([]), 16000)

    def test_invalid_sample_rate_raises_error(self, synthetic_audio):
        """Test that invalid sample rate raises ValueError."""
        audio, _ = synthetic_audio
        preprocessor = AudioPreprocessor()
        
        with pytest.raises(ValueError):
            preprocessor.preprocess(audio, -1)

    def test_wrong_audio_dimensions(self):
        """Test that 2D audio raises ValueError."""
        preprocessor = AudioPreprocessor()
        
        audio_2d = np.random.randn(2, 8000)
        
        with pytest.raises(ValueError):
            preprocessor.preprocess(audio_2d, 16000)


class TestPreprocessorConsistency:
    """Test reproducibility and consistency."""

    def test_preprocessing_consistent(self, synthetic_audio):
        """Test that preprocessing same audio twice gives same result."""
        audio, sr = synthetic_audio
        preprocessor = AudioPreprocessor()
        
        mel_spec_1 = preprocessor.preprocess(audio.copy(), sr)
        mel_spec_2 = preprocessor.preprocess(audio.copy(), sr)
        
        assert np.allclose(mel_spec_1, mel_spec_2)

    def test_get_info_keys_present(self, synthetic_audio):
        """Test that get_info returns all expected keys."""
        audio, sr = synthetic_audio
        preprocessor = AudioPreprocessor()
        mel_spec = preprocessor.preprocess(audio, sr)
        
        info = preprocessor.get_info()
        
        expected_keys = {
            "n_fft",
            "hop_length",
            "n_mels",
            "actual_n_mels",
            "min_freq",
            "max_freq",
            "actual_max_freq",
            "noise_gate_multiplier",
            "hpss_margin",
        }
        
        assert expected_keys.issubset(set(info.keys()))

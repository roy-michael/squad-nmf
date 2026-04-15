"""
Test suite for end-to-end pipeline integration.

Tests the complete audio-to-classification workflow.
"""
import pytest
import numpy as np
from preprocessor import AudioPreprocessor
from feature_extractor import NMFFeatureExtractor
from classifier import SoundClassifier, AudioClassificationPipeline


class TestPipelineIntegration:
    """Test complete pipeline integration."""

    def test_pipeline_audio_to_spectrogram(self, synthetic_audio):
        """Test audio preprocessing through pipeline."""
        audio, sr = synthetic_audio
        preprocessor = AudioPreprocessor()
        
        mel_spec = preprocessor.preprocess(audio, sr)
        
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] > 0
        assert mel_spec.shape[1] > 0

    def test_pipeline_spectrogram_to_features(self, mel_spectrogram):
        """Test feature extraction through pipeline."""
        feature_extractor = NMFFeatureExtractor(n_components=4)
        
        features = feature_extractor.extract(mel_spectrogram)
        
        assert len(features) == 4 * 8
        assert np.all(features >= 0)
        assert np.all(features <= 1)

    def test_pipeline_features_to_classification(self, multiple_feature_vectors):
        """Test classification through pipeline."""
        features, labels = multiple_feature_vectors
        classifier = SoundClassifier(model_type="rf")
        
        classifier.train(features, labels)
        predictions = classifier.predict(features[:5])
        
        assert len(predictions) == 5


class TestPipelineComponents:
    """Test integration of components."""

    def test_pipeline_end_to_end_flow(self, synthetic_audio):
        """Test complete flow: audio -> spec -> features -> classification."""
        audio, sr = synthetic_audio
        
        preprocessor = AudioPreprocessor()
        mel_spec = preprocessor.preprocess(audio, sr)
        
        feature_extractor = NMFFeatureExtractor(n_components=4)
        features = feature_extractor.extract(mel_spec)
        
        # Features should be ready for classification
        assert len(features) == 4 * 8
        assert np.all(features >= 0)
        assert np.all(features <= 1)

    def test_pipeline_multiple_audio_processing(self, synthetic_audio):
        """Test processing multiple audio files through pipeline."""
        audio, sr = synthetic_audio
        
        preprocessor = AudioPreprocessor()
        feature_extractor = NMFFeatureExtractor(n_components=4)
        
        feature_list = []
        for _ in range(3):
            mel_spec = preprocessor.preprocess(audio, sr)
            features = feature_extractor.extract(mel_spec)
            feature_list.append(features)
        
        features_array = np.array(feature_list)
        assert features_array.shape == (3, 4 * 8)


class TestPipelineEdgeCases:
    """Test pipeline with edge cases."""

    def test_pipeline_short_audio(self, short_audio):
        """Test pipeline with short audio."""
        audio, sr = short_audio
        
        preprocessor = AudioPreprocessor()
        mel_spec = preprocessor.preprocess(audio, sr)
        
        feature_extractor = NMFFeatureExtractor(n_components=2)
        features = feature_extractor.extract(mel_spec)
        
        assert len(features) == 2 * 8

    def test_pipeline_with_different_sample_rates(self):
        """Test pipeline robustness to different sample rates."""
        np.random.seed(42)
        
        for sr in [16000, 22050, 44100]:
            duration = 1.0
            audio = np.random.randn(int(sr * duration)).astype(np.float32)
            
            preprocessor = AudioPreprocessor()
            mel_spec = preprocessor.preprocess(audio, sr)
            
            assert mel_spec.shape[1] > 0


class TestPipelineReproducibility:
    """Test pipeline determinism and reproducibility."""

    def test_pipeline_reproducible_results(self, synthetic_audio):
        """Test that pipeline produces same results on same input."""
        audio, sr = synthetic_audio
        
        # First pass
        preprocessor1 = AudioPreprocessor()
        mel_spec1 = preprocessor1.preprocess(audio.copy(), sr)
        
        feature_extractor1 = NMFFeatureExtractor(n_components=4)
        features1 = feature_extractor1.extract(mel_spec1)
        
        # Second pass
        preprocessor2 = AudioPreprocessor()
        mel_spec2 = preprocessor2.preprocess(audio.copy(), sr)
        
        feature_extractor2 = NMFFeatureExtractor(n_components=4)
        features2 = feature_extractor2.extract(mel_spec2)
        
        # Should be close
        assert np.allclose(features1, features2, rtol=1e-4)


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_workflow(self, synthetic_audio, multiple_feature_vectors):
        """Test complete workflow: preprocess -> extract -> classify."""
        audio, sr = synthetic_audio
        features, labels = multiple_feature_vectors  # 48-dim features
        
        # Setup pipeline
        preprocessor = AudioPreprocessor()
        feature_extractor = NMFFeatureExtractor(n_components=6)  # Must match default
        classifier = SoundClassifier(model_type="rf")
        
        # Process audio
        mel_spec = preprocessor.preprocess(audio, sr)
        features_extracted = feature_extractor.extract(mel_spec)
        
        # Train on synthetic features
        classifier.train(features, labels)
        
        # Predict
        prediction = classifier.predict(features_extracted.reshape(1, -1))
        
        assert prediction.shape == (1,)
        assert prediction[0] in np.unique(labels)

"""
End-to-end pipeline orchestration and configuration for underwater vehicle sound classification.

This module provides the UnderWaterAudioPipeline class that orchestrates all
components (audio loading, preprocessing, feature extraction, and classification)
into a cohesive pipeline with configuration management, logging, and error handling.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import yaml

from audio_loader import load_audio
from preprocessor import AudioPreprocessor
from feature_extractor import NMFFeatureExtractor
from classifier import SoundClassifier, ClassifierDataset


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class UnderWaterAudioPipeline:
    """
    Comprehensive end-to-end pipeline orchestrator for underwater audio classification.

    Integrates audio loading, preprocessing, NMF feature extraction, and ML-based
    classification with full configuration management and error handling.

    Attributes
    ----------
    loader : callable
        Audio loading function (load_audio)
    preprocessor : AudioPreprocessor
        Audio preprocessing module
    feature_extractor : NMFFeatureExtractor
        NMF-based feature extraction module
    classifier : SoundClassifier
        Machine learning classifier
    config : Dict
        Complete pipeline configuration
    label_names : Dict[int, str]
        Mapping from class IDs to human-readable labels
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        preprocessor_config: Optional[Dict] = None,
        nmf_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        verbose: bool = False,
    ):
        """
        Initialize the underwater audio pipeline with optional configuration overrides.

        Parameters
        ----------
        config : Dict, optional
            Complete pipeline configuration dict. If provided, other config
            parameters are ignored.
        preprocessor_config : Dict, optional
            Configuration for AudioPreprocessor (overrides config['preprocessor'])
        nmf_config : Dict, optional
            Configuration for NMFFeatureExtractor (overrides config['nmf'])
        classifier_config : Dict, optional
            Configuration for SoundClassifier (overrides config['classifier'])
        verbose : bool, optional
            Enable verbose logging output (default: False)

        Raises
        ------
        ValueError
            If configuration is invalid or missing required keys.
        """
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Load or initialize configuration
        if config is None:
            config = self._get_default_config()
            logger.info("Using default pipeline configuration")
        else:
            logger.info("Using provided pipeline configuration")

        # Apply configuration overrides
        if preprocessor_config is not None:
            config['preprocessor'].update(preprocessor_config)
        if nmf_config is not None:
            config['nmf'].update(nmf_config)
        if classifier_config is not None:
            config['classifier'].update(classifier_config)

        self.config = config
        self.label_names: Dict[int, str] = config.get('label_names', {})

        logger.debug(f"Pipeline config: {config}")

        # Initialize components
        try:
            self._initialize_components()
            logger.info("Pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """
        Get default pipeline configuration.

        Returns
        -------
        config : Dict
            Default configuration dictionary
        """
        return {
            'preprocessor': {
                'n_fft': 8192,
                'min_freq': 200,
                'max_freq': 12000,
                'n_mels': 512,
                'noise_gate_multiplier': 1.5,
                'hpss_margin': 3.0,
            },
            'nmf': {
                'n_components': 6,
                'use_sklearn': True,
                'max_iter': 500,
            },
            'classifier': {
                'model_type': 'rf',
                'random_state': 42,
            },
            'label_names': {
                '0': 'AUV',
                '1': 'Torpedo',
                '2': 'Surface_Vessel',
            }
        }

    def _initialize_components(self) -> None:
        """
        Initialize all pipeline components from configuration.

        Raises
        ------
        ValueError
            If component initialization fails.
        """
        # Initialize audio loader
        self.loader = load_audio
        logger.debug("Audio loader initialized")

        # Initialize preprocessor
        preprocessor_config = self.config.get('preprocessor', {})
        self.preprocessor = AudioPreprocessor(
            n_fft=preprocessor_config.get('n_fft', 8192),
            min_freq=preprocessor_config.get('min_freq', 200),
            max_freq=preprocessor_config.get('max_freq', 12000),
            n_mels=preprocessor_config.get('n_mels', 512),
            noise_gate_multiplier=preprocessor_config.get('noise_gate_multiplier', 1.5),
            hpss_margin=preprocessor_config.get('hpss_margin', 3.0),
        )
        logger.debug("AudioPreprocessor initialized")

        # Initialize feature extractor
        nmf_config = self.config.get('nmf', {})
        self.feature_extractor = NMFFeatureExtractor(
            n_components=nmf_config.get('n_components', 6),
            use_sklearn=nmf_config.get('use_sklearn', True),
            max_iter=nmf_config.get('max_iter', 500),
        )
        logger.debug("NMFFeatureExtractor initialized")

        # Initialize classifier
        classifier_config = self.config.get('classifier', {})
        self.classifier = SoundClassifier(
            model_type=classifier_config.get('model_type', 'rf'),
            random_state=classifier_config.get('random_state', 42),
        )
        logger.debug("SoundClassifier initialized")

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Process a single audio file through the complete preprocessing pipeline.

        Loads audio, applies preprocessing, and extracts features without
        performing classification.

        Parameters
        ----------
        audio_path : str
            Path to the audio file (WAV format)

        Returns
        -------
        result : Dict
            Dictionary containing:
            - 'features': Feature vector (n_components*8,) normalized to [0, 1]
            - 'spectrogram': Mel spectrogram (n_mels, n_frames) for visualization
            - 'audio_info': Dict with 'sample_rate', 'duration', 'num_samples', 'file_path'
            - 'preprocessor_info': Dict with preprocessing parameters applied

        Raises
        ------
        FileNotFoundError
            If audio file does not exist
        ValueError
            If audio file cannot be loaded or is invalid
        """
        logger.info(f"Processing audio file: {audio_path}")

        # Validate file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio
            logger.debug(f"Loading audio from {audio_path}")
            audio_data, sample_rate = self.loader(audio_path)
            duration = len(audio_data) / sample_rate

            logger.debug(
                f"Audio loaded: {len(audio_data)} samples at {sample_rate} Hz "
                f"({duration:.2f}s)"
            )

            # Preprocess audio
            logger.debug("Preprocessing audio")
            mel_spectrogram = self.preprocessor.preprocess(audio_data, sample_rate)

            # Extract features
            logger.debug("Extracting features via NMF")
            features = self.feature_extractor.extract(mel_spectrogram)

            logger.info(
                f"Audio processing complete: {len(features)} features extracted"
            )

            return {
                'features': features,
                'spectrogram': mel_spectrogram,
                'audio_info': {
                    'file_path': os.path.abspath(audio_path),
                    'sample_rate': int(sample_rate),
                    'num_samples': len(audio_data),
                    'duration_seconds': float(duration),
                },
                'preprocessor_info': self.preprocessor.get_info(),
            }

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            raise

    def train(
        self,
        audio_paths: List[str],
        labels: List[int],
        test_size: float = 0.2,
        save_model: bool = True,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the classifier on a batch of audio files.

        Processes all audio files through the complete pipeline, extracts features,
        and trains the ML classifier with stratified train/test split.

        Parameters
        ----------
        audio_paths : List[str]
            List of paths to audio files
        labels : List[int]
            Corresponding class labels (0-indexed)
        test_size : float, optional
            Proportion of data for testing (default: 0.2, range: 0.0-1.0)
        save_model : bool, optional
            Save trained model to disk (default: True)
        model_path : str, optional
            Path to save model. If None and save_model=True, saves to
            'underwater_sound_classifier.joblib' in current directory

        Returns
        -------
        result : Dict
            Dictionary containing:
            - 'n_samples': Total number of audio files processed
            - 'train_metrics': Dict with accuracy, precision, recall, f1 on training set
            - 'test_metrics': Dict with accuracy, precision, recall, f1 on test set
            - 'model_path': Path where model was saved (if save_model=True)
            - 'label_names': Dict mapping class ID to label name

        Raises
        ------
        ValueError
            If audio_paths and labels have different lengths or are empty
        FileNotFoundError
            If any audio file does not exist
        Exception
            If audio processing fails
        """
        if not audio_paths:
            logger.error("audio_paths list is empty")
            raise ValueError("audio_paths cannot be empty")

        if len(audio_paths) != len(labels):
            logger.error(
                f"Mismatch: {len(audio_paths)} audio files but {len(labels)} labels"
            )
            raise ValueError(
                f"Length mismatch: {len(audio_paths)} audio files but {len(labels)} labels"
            )

        logger.info(f"Starting training on {len(audio_paths)} audio files")

        try:
            # Process all audio files
            features_list: List[np.ndarray] = []

            for idx, (audio_path, label) in enumerate(zip(audio_paths, labels)):
                logger.debug(f"Processing {idx + 1}/{len(audio_paths)}: {audio_path}")

                result = self.process_audio(audio_path)
                features_list.append(result['features'])

            logger.info(f"Feature extraction complete for all {len(features_list)} files")

            # Create dataset and perform stratified split
            logger.debug("Creating dataset and performing train/test split")
            dataset = ClassifierDataset(features_list, labels)
            self.label_names = dataset.get_label_names()

            train_data, test_data = dataset.train_test_split(
                test_size=test_size,
                random_state=42,
            )

            logger.info(
                f"Train set: {len(train_data['features'])} samples, "
                f"Test set: {len(test_data['features'])} samples"
            )

            # Train classifier
            logger.debug("Training classifier")
            self.classifier.train(
                train_data['features'],
                train_data['labels'],
            )

            # Evaluate on both sets
            logger.debug("Evaluating classifier on train and test sets")
            train_metrics = self.classifier.evaluate(
                train_data['features'],
                train_data['labels'],
            )
            test_metrics = self.classifier.evaluate(
                test_data['features'],
                test_data['labels'],
            )

            logger.info(
                f"Training complete - Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Test Acc: {test_metrics['accuracy']:.4f}"
            )

            # Save model if requested
            saved_path = None
            if save_model:
                if model_path is None:
                    model_path = 'underwater_sound_classifier.joblib'

                logger.info(f"Saving model to {model_path}")
                self.classifier.save(model_path)
                saved_path = os.path.abspath(model_path)

            return {
                'n_samples': len(features_list),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model_path': saved_path,
                'label_names': self.label_names,
            }

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise

    def classify(self, audio_path: str) -> Dict[str, Any]:
        """
        Classify a single audio file using the trained classifier.

        Requires a trained model. Use train() method or load_config() first.

        Parameters
        ----------
        audio_path : str
            Path to the audio file (WAV format)

        Returns
        -------
        result : Dict
            Dictionary containing:
            - 'file': Input audio file path
            - 'predicted_class': Predicted class ID (int)
            - 'predicted_label': Human-readable class name (str)
            - 'confidence': Confidence score (float, range [0, 1])
            - 'probabilities': Dict mapping class_id (str) to probability (float)
            - 'audio_info': Dict with sample_rate, duration, num_samples

        Raises
        ------
        RuntimeError
            If classifier has not been trained
        FileNotFoundError
            If audio file does not exist
        ValueError
            If audio file is invalid
        """
        if not self.classifier.is_trained:
            logger.error("Classifier has not been trained")
            raise RuntimeError(
                "Classifier must be trained before classification. "
                "Use train() method or load trained model first."
            )

        logger.info(f"Classifying audio file: {audio_path}")

        try:
            # Process audio
            result = self.process_audio(audio_path)
            features = result['features']

            # Predict class and probabilities
            logger.debug("Predicting class")
            predicted_class = self.classifier.predict(features.reshape(1, -1))[0]
            probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]

            # Get label
            predicted_label = self.label_names.get(
                int(predicted_class),
                f"Class_{predicted_class}"
            )
            confidence = float(np.max(probabilities))

            # Map probabilities to class labels
            prob_dict = {
                str(class_id): float(probabilities[i])
                for i, class_id in enumerate(sorted(self.label_names.keys()))
            }

            logger.info(
                f"Classification complete: {predicted_label} "
                f"(confidence: {confidence:.4f})"
            )

            return {
                'file': os.path.abspath(audio_path),
                'predicted_class': int(predicted_class),
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': prob_dict,
                'audio_info': result['audio_info'],
            }

        except Exception as e:
            logger.error(f"Error during classification: {e}", exc_info=True)
            raise

    def batch_classify(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple audio files in batch.

        Parameters
        ----------
        audio_paths : List[str]
            List of paths to audio files

        Returns
        -------
        results : List[Dict]
            List of classification results from classify() for each file

        Raises
        ------
        RuntimeError
            If classifier has not been trained
        ValueError
            If audio_paths is empty
        """
        if not audio_paths:
            logger.error("audio_paths list is empty")
            raise ValueError("audio_paths cannot be empty")

        if not self.classifier.is_trained:
            logger.error("Classifier has not been trained")
            raise RuntimeError("Classifier must be trained before classification")

        logger.info(f"Batch classifying {len(audio_paths)} audio files")

        results = []
        for idx, audio_path in enumerate(audio_paths):
            logger.debug(f"Classifying {idx + 1}/{len(audio_paths)}: {audio_path}")
            try:
                result = self.classify(audio_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to classify {audio_path}: {e}")
                results.append({
                    'file': audio_path,
                    'error': str(e),
                    'predicted_class': None,
                    'predicted_label': None,
                })

        logger.info(f"Batch classification complete: {len(results)} files processed")
        return results

    def save_config(self, filepath: str) -> None:
        """
        Save pipeline configuration to YAML or JSON file.

        File format is determined by extension (.yaml/.yml for YAML, .json for JSON).

        Parameters
        ----------
        filepath : str
            Path to save configuration file

        Raises
        ------
        ValueError
            If file extension is not supported
        IOError
            If write fails
        """
        logger.info(f"Saving pipeline configuration to {filepath}")

        config_to_save = {
            'preprocessor': self.config['preprocessor'].copy(),
            'nmf': self.config['nmf'].copy(),
            'classifier': self.config['classifier'].copy(),
            'label_names': self.label_names,
        }

        try:
            filepath_lower = filepath.lower()

            if filepath_lower.endswith(('.yaml', '.yml')):
                with open(filepath, 'w') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False)
                logger.info(f"Configuration saved to {filepath} (YAML format)")

            elif filepath_lower.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(config_to_save, f, indent=2)
                logger.info(f"Configuration saved to {filepath} (JSON format)")

            else:
                logger.error(f"Unsupported file format: {filepath}")
                raise ValueError(
                    "File extension must be .yaml, .yml, or .json"
                )

        except Exception as e:
            logger.error(f"Error saving configuration: {e}", exc_info=True)
            raise

    @classmethod
    def load_config(cls, filepath: str) -> 'UnderWaterAudioPipeline':
        """
        Create pipeline instance from saved configuration file.

        File format is determined by extension (.yaml/.yml for YAML, .json for JSON).

        Parameters
        ----------
        filepath : str
            Path to configuration file

        Returns
        -------
        pipeline : UnderWaterAudioPipeline
            Initialized pipeline with loaded configuration

        Raises
        ------
        FileNotFoundError
            If configuration file does not exist
        ValueError
            If file format is not supported or config is invalid
        """
        logger.info(f"Loading pipeline configuration from {filepath}")

        if not os.path.exists(filepath):
            logger.error(f"Configuration file not found: {filepath}")
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            filepath_lower = filepath.lower()

            if filepath_lower.endswith(('.yaml', '.yml')):
                with open(filepath, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {filepath} (YAML format)")

            elif filepath_lower.endswith('.json'):
                with open(filepath, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {filepath} (JSON format)")

            else:
                logger.error(f"Unsupported file format: {filepath}")
                raise ValueError(
                    "File extension must be .yaml, .yml, or .json"
                )

            return cls(config=config)

        except Exception as e:
            logger.error(f"Error loading configuration: {e}", exc_info=True)
            raise

    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained classifier model.

        Parameters
        ----------
        model_path : str
            Path to saved model file (joblib format)

        Raises
        ------
        FileNotFoundError
            If model file does not exist
        Exception
            If model loading fails
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading trained model from {model_path}")

        try:
            self.classifier = SoundClassifier.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about pipeline state and configuration.

        Returns
        -------
        info : Dict
            Dictionary containing:
            - 'classifier_trained': bool indicating if classifier is trained
            - 'n_components': Number of NMF components
            - 'model_type': Type of classifier (rf, svm, gb)
            - 'label_names': Mapping of class IDs to names
            - 'config': Complete pipeline configuration
        """
        return {
            'classifier_trained': self.classifier.is_trained,
            'n_components': self.feature_extractor.n_components,
            'model_type': self.classifier.model_type,
            'label_names': self.label_names,
            'config': self.config,
        }

#!/usr/bin/env python3
"""
Demo: Train classifier on ocean recordings from Croatia dataset.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio_loader import load_audio
from src.preprocessor import AudioPreprocessor
from src.feature_extractor import NMFFeatureExtractor
from src.classifier import ClassifierDataset, SoundClassifier
from src.pipeline import UnderWaterAudioPipeline

def main():
    """Train classifier on ocean recordings."""
    
    # Paths
    data_dir = Path('./data')
    model_dir = Path('./models')
    model_path = model_dir / 'ocean_classifier.pkl'
    config_path = model_dir / 'config.yaml'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files organized by class (subdirectory = class)
    logger.info(f"Loading audio files from: {data_dir}")
    
    audio_files = []
    labels = []
    label_to_id = {}
    next_id = 0
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        if class_name not in label_to_id:
            label_to_id[class_name] = next_id
            next_id += 1
        
        class_id = label_to_id[class_name]
        
        wav_files = list(class_dir.glob('*.wav'))
        logger.info(f"Found {len(wav_files)} files in class '{class_name}' (ID: {class_id})")
        
        for wav_file in wav_files:
            audio_files.append(str(wav_file))
            labels.append(class_id)
    
    logger.info(f"Total files: {len(audio_files)}, Classes: {len(label_to_id)}")
    logger.info(f"Label mapping: {label_to_id}")
    
    # Load audio and extract features
    logger.info("Loading and preprocessing audio files...")
    
    features_list = []
    valid_labels = []
    
    preprocessor = AudioPreprocessor(n_fft=8192, min_freq=200, max_freq=12000, n_mels=512)
    feature_extractor = NMFFeatureExtractor(n_components=6, use_sklearn=True, max_iter=500)
    
    for i, (audio_path, label) in enumerate(zip(audio_files, labels)):
        try:
            logger.info(f"[{i+1}/{len(audio_files)}] Processing: {Path(audio_path).name}")
            
            # Load
            y, sr = load_audio(audio_path)
            
            # Preprocess
            S_mel = preprocessor.preprocess(y, sr)
            
            # Extract features
            features = feature_extractor.extract(S_mel)
            
            features_list.append(features)
            valid_labels.append(label)
            
            logger.info(f"  ✓ Features shape: {features.shape}")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {audio_path}: {e}")
            continue
    
    if not features_list:
        logger.error("No features extracted. Exiting.")
        return
    
    logger.info(f"Successfully extracted features from {len(features_list)} files")
    
    # Train classifier
    logger.info("Training classifier...")
    
    dataset = ClassifierDataset(features_list, valid_labels)
    train_data, test_data = dataset.train_test_split(test_size=0.2, random_state=42)
    
    classifier = SoundClassifier(model_type='rf', random_state=42)
    classifier.train(train_data['features'], train_data['labels'])
    
    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = classifier.evaluate(test_data['features'], test_data['labels'])
    
    logger.info("=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name:20s}: {value:.4f}")
    logger.info("=" * 60)
    
    # Save model
    logger.info(f"Saving model to: {model_path}")
    classifier.save(str(model_path))
    
    # Test classification
    logger.info("\nTesting classification on first test sample...")
    test_features = test_data['features'][0:1]
    pred_label = classifier.predict(test_features)[0]
    pred_proba = classifier.predict_proba(test_features)[0]
    
    # Reverse label mapping
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    logger.info(f"Predicted class ID: {pred_label}")
    logger.info(f"Predicted class name: {id_to_label[pred_label]}")
    logger.info(f"Confidence: {pred_proba[pred_label]:.2%}")
    logger.info("\nAll class probabilities:")
    for class_id, prob in enumerate(pred_proba):
        logger.info(f"  {id_to_label[class_id]:20s}: {prob:.2%}")
    
    logger.info("\n✅ Training complete! Model saved to: " + str(model_path))

if __name__ == '__main__':
    main()

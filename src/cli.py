"""
Professional command-line interface for underwater vehicle sound classification.

Provides commands for training models, classifying audio files, batch processing,
audio analysis, and configuration management.
"""

import os
import sys
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

import click
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from audio_loader import load_audio
from preprocessor import AudioPreprocessor
from feature_extractor import NMFFeatureExtractor
from classifier import SoundClassifier, ClassifierDataset
from pipeline import UnderWaterAudioPipeline


# ============================================================================
# Helper Functions
# ============================================================================

def load_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default config.

    Returns
    -------
    config : dict
        Pipeline configuration dictionary.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        pipeline = UnderWaterAudioPipeline()
        return pipeline.config


def save_pipeline_config(config: Dict[str, Any], filepath: str) -> None:
    """Save pipeline configuration to YAML file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def find_config_file(model_dir: str) -> Optional[str]:
    """
    Find config file in model directory.

    Parameters
    ----------
    model_dir : str
        Directory containing model file.

    Returns
    -------
    config_path : str or None
        Path to config.yaml if found, None otherwise.
    """
    config_path = os.path.join(model_dir, 'config.yaml')
    if os.path.exists(config_path):
        return config_path
    return None


def validate_audio_dir(data_dir: str) -> Dict[str, List[str]]:
    """
    Scan data directory for WAV files organized by subdirectories (classes).

    Parameters
    ----------
    data_dir : str
        Root directory containing subdirectories with class audio files.

    Returns
    -------
    class_files : dict
        Mapping of class name to list of audio file paths.

    Raises
    ------
    ValueError
        If no subdirectories or audio files found.
    """
    class_files = {}

    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]

    if not subdirs:
        raise ValueError(f"No subdirectories found in {data_dir}")

    for class_name in subdirs:
        class_dir = os.path.join(data_dir, class_name)
        audio_files = [os.path.join(class_dir, f)
                      for f in os.listdir(class_dir)
                      if f.lower().endswith(('.wav', '.mp3', '.flac'))]

        if audio_files:
            class_files[class_name] = audio_files

    if not class_files:
        raise ValueError(f"No audio files found in subdirectories of {data_dir}")

    return class_files


def get_label_mapping(class_names: List[str]) -> Dict[str, int]:
    """Create numeric label mapping from class names."""
    return {name: i for i, name in enumerate(sorted(class_names))}


def format_metrics(metrics: Dict[str, float], decimals: int = 4) -> str:
    """Format metrics dictionary as human-readable string."""
    lines = []
    for key, value in metrics.items():
        lines.append(f"  {key:.<20} {value:.{decimals}f}")
    return '\n'.join(lines)


# ============================================================================
# Click Command Group
# ============================================================================

@click.group()
@click.version_option(version='1.0.0', prog_name='underwater-audio')
def main() -> None:
    """
    Underwater Vehicle Sound Classifier - Professional CLI

    This tool provides command-line access to train, classify, and analyze
    underwater vehicle sounds using machine learning and NMF features.

    Examples:
        underwater-audio train --data-dir ./data --model-type rf
        underwater-audio classify --audio-file ./sample.wav --model ./models/classifier.pkl
        underwater-audio batch-classify --input-dir ./test_audio --model ./models/classifier.pkl
        underwater-audio analyze --audio-file ./sample.wav
    """
    pass


# ============================================================================
# Train Command
# ============================================================================

@main.command()
@click.option(
    '--data-dir',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Directory containing training WAV files in subdirectories (one per class).'
)
@click.option(
    '--model-type',
    type=click.Choice(['rf', 'svm', 'gb'], case_sensitive=False),
    default='rf',
    help='Model type: Random Forest (rf), SVM (svm), or Gradient Boosting (gb). [default: rf]'
)
@click.option(
    '--output-model',
    type=click.Path(),
    default='./models/classifier.pkl',
    help='Path to save trained model. [default: ./models/classifier.pkl]'
)
@click.option(
    '--test-size',
    type=float,
    default=0.2,
    help='Train/test split ratio. [default: 0.2]'
)
@click.option(
    '--n-components',
    type=int,
    default=6,
    help='Number of NMF components. [default: 6]'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose logging.'
)
def train(
    data_dir: str,
    model_type: str,
    output_model: str,
    test_size: float,
    n_components: int,
    verbose: bool
) -> None:
    """
    Train a classifier on underwater audio data.

    Loads all WAV files from subdirectories (each subdir = class label),
    extracts features using NMF, and trains a machine learning model.

    Example:
        underwater-audio train --data-dir ./data --model-type rf \\
            --output-model ./models/classifier.pkl --verbose
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        click.echo(click.style("[INFO] Verbose mode enabled", fg="cyan"))

    try:
        # Validate data directory
        click.echo(click.style("[*] Scanning data directory...", fg="cyan"))
        class_files = validate_audio_dir(data_dir)
        click.echo(f"    Found {len(class_files)} classes: {', '.join(sorted(class_files.keys()))}")

        # Create label mapping
        label_mapping = get_label_mapping(list(class_files.keys()))
        total_files = sum(len(files) for files in class_files.values())
        click.echo(f"    Total audio files: {total_files}")

        # Prepare audio paths and labels
        audio_paths = []
        labels = []
        for class_name in sorted(class_files.keys()):
            for audio_file in class_files[class_name]:
                audio_paths.append(audio_file)
                labels.append(label_mapping[class_name])

        # Initialize pipeline with custom config
        click.echo(click.style("[*] Initializing pipeline...", fg="cyan"))
        pipeline = UnderWaterAudioPipeline(verbose=verbose)
        pipeline.feature_extractor.n_components = n_components

        # Train pipeline
        click.echo(click.style("[*] Training model...", fg="cyan"))
        with click.progressbar(
            length=len(audio_paths),
            label="    Processing audio files",
            show_pos=True
        ) as pbar:
            features_list = []
            valid_labels = []
            for idx, audio_path in enumerate(audio_paths):
                try:
                    audio_data, sr = load_audio(audio_path)
                    mel_spec = pipeline.preprocessor.preprocess(audio_data, sr)
                    features = pipeline.feature_extractor.extract(mel_spec)
                    features_list.append(features)
                    valid_labels.append(labels[idx])
                    pbar.update(1)
                except Exception as e:
                    if verbose:
                        click.echo(f"\n    Warning: Failed to process {audio_path}: {str(e)}", err=True)
                    pbar.update(1)

        # Create dataset and split
        dataset = ClassifierDataset(features_list, valid_labels)
        train_data, test_data = dataset.train_test_split(
            test_size=test_size, random_state=42
        )

        # Train classifier
        classifier = SoundClassifier(model_type=model_type)
        classifier.train(
            train_data['features'],
            train_data['labels'],
        )
        pipeline.classifier = classifier
        pipeline.label_names = dataset.get_label_names()

        # Evaluate
        click.echo(click.style("\n[*] Evaluating model...", fg="cyan"))
        train_metrics = classifier.evaluate(
            train_data['features'],
            train_data['labels']
        )
        test_metrics = classifier.evaluate(
            test_data['features'],
            test_data['labels']
        )

        # Display metrics
        click.echo(click.style("\n[✓] Training Metrics:", fg="green"))
        click.echo(format_metrics(train_metrics))

        click.echo(click.style("\n[✓] Test Metrics:", fg="green"))
        click.echo(format_metrics(test_metrics))

        # Save model and config
        click.echo(click.style("\n[*] Saving model...", fg="cyan"))
        os.makedirs(os.path.dirname(output_model) or '.', exist_ok=True)
        classifier.save(output_model)

        # Save config
        config_path = os.path.join(
            os.path.dirname(output_model) or '.',
            'config.yaml'
        )
        pipeline.config['nmf']['n_components'] = n_components
        pipeline.config['classifier']['model_type'] = model_type
        save_pipeline_config(pipeline.config, config_path)

        click.echo(f"    Model saved to: {click.style(output_model, fg='green')}")
        click.echo(f"    Config saved to: {click.style(config_path, fg='green')}")

        click.echo(click.style("\n[✓] Training completed successfully!", fg="green"))

    except Exception as e:
        click.echo(click.style(f"\n[✗] Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


# ============================================================================
# Classify Command
# ============================================================================

@main.command()
@click.option(
    '--audio-file',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='Path to WAV file to classify.'
)
@click.option(
    '--model',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='Path to trained classifier model.'
)
@click.option(
    '--config',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    help='Path to config YAML (auto-detected if in model dir).'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed output.'
)
@click.option(
    '--save-json',
    type=click.Path(),
    default=None,
    help='Save results to JSON file.'
)
def classify(
    audio_file: str,
    model: str,
    config: Optional[str],
    verbose: bool,
    save_json: Optional[str]
) -> None:
    """
    Classify a single audio file using a trained model.

    Example:
        underwater-audio classify --audio-file ./sample.wav \\
            --model ./models/classifier.pkl --verbose
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        click.echo(click.style("[INFO] Verbose mode enabled", fg="cyan"))

    try:
        # Auto-detect config if not provided
        if config is None:
            model_dir = os.path.dirname(model) or '.'
            detected_config = find_config_file(model_dir)
            if detected_config:
                config = detected_config
                click.echo(f"    Auto-detected config: {config}")

        # Load configuration
        click.echo(click.style("[*] Loading configuration...", fg="cyan"))
        pipeline_config = load_pipeline_config(config)

        # Initialize pipeline
        pipeline = UnderWaterAudioPipeline(config=pipeline_config, verbose=verbose)

        # Load model
        click.echo(click.style("[*] Loading model...", fg="cyan"))
        pipeline.classifier = SoundClassifier.load(model)

        # Classify
        click.echo(click.style("[*] Classifying audio...", fg="cyan"))
        result = pipeline.classify(audio_file)

        # Display results
        click.echo(click.style("\n[✓] Classification Results:", fg="green"))
        click.echo(f"  {'Audio file:':.<30} {audio_file}")
        click.echo(f"  {'Predicted class:':.<30} {result['predicted_label']}")
        click.echo(f"  {'Confidence:':.<30} {result['confidence']:.4f}")

        if verbose:
            click.echo(click.style("\n[*] Class Probabilities:", fg="cyan"))
            for idx, prob in enumerate(result['probabilities']):
                click.echo(f"    Class {idx}: {prob:.4f}")

        # Save to JSON if requested
        if save_json:
            os.makedirs(os.path.dirname(save_json) or '.', exist_ok=True)
            output_data = {
                'audio_file': audio_file,
                'predicted_class': result['predicted_class'],
                'predicted_label': result['predicted_label'],
                'confidence': float(result['confidence']),
                'probabilities': [float(p) for p in result['probabilities']]
            }
            with open(save_json, 'w') as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"\n    Results saved to: {click.style(save_json, fg='green')}")

        click.echo(click.style("\n[✓] Classification completed!", fg="green"))

    except Exception as e:
        click.echo(click.style(f"\n[✗] Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


# ============================================================================
# Batch Classify Command
# ============================================================================

@main.command()
@click.option(
    '--input-dir',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Directory with WAV files to classify.'
)
@click.option(
    '--model',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='Path to trained classifier model.'
)
@click.option(
    '--config',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    help='Path to config YAML (auto-detected if in model dir).'
)
@click.option(
    '--output-csv',
    type=click.Path(),
    default=None,
    help='Save results to CSV file.'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show progress.'
)
def batch_classify(
    input_dir: str,
    model: str,
    config: Optional[str],
    output_csv: Optional[str],
    verbose: bool
) -> None:
    """
    Classify all audio files in a directory.

    Example:
        underwater-audio batch-classify --input-dir ./test_audio \\
            --model ./models/classifier.pkl --output-csv results.csv
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Find audio files
        click.echo(click.style("[*] Scanning input directory...", fg="cyan"))
        audio_files = [os.path.join(input_dir, f)
                      for f in os.listdir(input_dir)
                      if f.lower().endswith(('.wav', '.mp3', '.flac'))]

        if not audio_files:
            raise ValueError(f"No audio files found in {input_dir}")

        click.echo(f"    Found {len(audio_files)} audio files")

        # Auto-detect config if not provided
        if config is None:
            model_dir = os.path.dirname(model) or '.'
            detected_config = find_config_file(model_dir)
            if detected_config:
                config = detected_config

        # Load configuration
        click.echo(click.style("[*] Loading configuration...", fg="cyan"))
        pipeline_config = load_pipeline_config(config)

        # Initialize pipeline
        pipeline = UnderWaterAudioPipeline(config=pipeline_config, verbose=verbose)

        # Load model
        click.echo(click.style("[*] Loading model...", fg="cyan"))
        pipeline.classifier = SoundClassifier.load(model)

        # Classify all files
        click.echo(click.style("[*] Classifying audio files...", fg="cyan"))
        results = []

        with click.progressbar(
            audio_files,
            label="    Processing",
            show_pos=True
        ) as pbar:
            for audio_file in pbar:
                try:
                    result = pipeline.classify(audio_file)
                    results.append({
                        'filename': os.path.basename(audio_file),
                        'filepath': audio_file,
                        'predicted_class': result['predicted_class'],
                        'predicted_label': result['predicted_label'],
                        'confidence': float(result['confidence']),
                    })
                except Exception as e:
                    if verbose:
                        click.echo(f"\n    Warning: Failed to classify {audio_file}: {str(e)}")
                    results.append({
                        'filename': os.path.basename(audio_file),
                        'filepath': audio_file,
                        'predicted_class': None,
                        'predicted_label': 'ERROR',
                        'confidence': 0.0,
                    })

        # Display summary
        successful = sum(1 for r in results if r['predicted_label'] != 'ERROR')
        click.echo(click.style("\n[✓] Batch Classification Results:", fg="green"))
        click.echo(f"  {'Total files:':.<30} {len(results)}")
        click.echo(f"  {'Successfully classified:':.<30} {successful}")
        click.echo(f"  {'Failed:':.<30} {len(results) - successful}")

        # Display class distribution
        if successful > 0:
            class_counts = {}
            for r in results:
                if r['predicted_label'] != 'ERROR':
                    label = r['predicted_label']
                    class_counts[label] = class_counts.get(label, 0) + 1

            click.echo(click.style("\n[*] Class Distribution:", fg="cyan"))
            for label, count in sorted(class_counts.items()):
                pct = (count / successful) * 100
                click.echo(f"    {label:.<20} {count:>3} ({pct:>5.1f}%)")

        # Save to CSV if requested
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['filename', 'filepath', 'predicted_class',
                               'predicted_label', 'confidence']
                )
                writer.writeheader()
                writer.writerows(results)
            click.echo(f"\n    Results saved to: {click.style(output_csv, fg='green')}")

        click.echo(click.style("\n[✓] Batch classification completed!", fg="green"))

    except Exception as e:
        click.echo(click.style(f"\n[✗] Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


# ============================================================================
# Analyze Command
# ============================================================================

@main.command()
@click.option(
    '--audio-file',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='Path to WAV file to analyze.'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='./diagnostics',
    help='Directory to save diagnostic plots. [default: ./diagnostics]'
)
@click.option(
    '--model',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    help='Trained model for predictions (optional).'
)
@click.option(
    '--config',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    help='Path to config YAML (auto-detected if in model dir).'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed output.'
)
def analyze(
    audio_file: str,
    output_dir: str,
    model: Optional[str],
    config: Optional[str],
    verbose: bool
) -> None:
    """
    Analyze audio file and generate diagnostic plots.

    Generates waveform, spectrogram, and NMF component visualizations.
    Optionally includes classification result if model is provided.

    Example:
        underwater-audio analyze --audio-file ./sample.wav \\
            --output-dir ./results --model ./models/classifier.pkl
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        click.echo(click.style("[INFO] Verbose mode enabled", fg="cyan"))

    try:
        # Load audio
        click.echo(click.style("[*] Loading audio file...", fg="cyan"))
        audio_data, sr = load_audio(audio_file)
        duration = len(audio_data) / sr

        click.echo(f"  {'Duration:':.<20} {duration:.2f} seconds")
        click.echo(f"  {'Sample rate:':.<20} {sr} Hz")
        click.echo(f"  {'Samples:':.<20} {len(audio_data):,}")

        # Initialize preprocessor and feature extractor
        click.echo(click.style("[*] Initializing analysis...", fg="cyan"))
        if config:
            pipeline_config = load_pipeline_config(config)
        else:
            pipeline = UnderWaterAudioPipeline()
            pipeline_config = pipeline.config

        preprocessor = AudioPreprocessor(
            **pipeline_config['preprocessor']
        )
        feature_extractor = NMFFeatureExtractor(
            **pipeline_config['nmf']
        )

        # Preprocess audio
        click.echo(click.style("[*] Preprocessing audio...", fg="cyan"))
        mel_spec = preprocessor.preprocess(audio_data, sr)
        click.echo(f"    Mel spectrogram shape: {mel_spec.shape}")

        # Extract features
        click.echo(click.style("[*] Extracting NMF features...", fg="cyan"))
        features = feature_extractor.extract(mel_spec)
        W, H = feature_extractor.get_components()
        click.echo(f"    NMF basis shape: {W.shape}")
        click.echo(f"    NMF activations shape: {H.shape}")
        click.echo(f"    Feature vector shape: {features.shape}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate diagnostic report
        click.echo(click.style("[*] Generating diagnostics...", fg="cyan"))

        report_lines = [
            "=" * 60,
            "Audio Analysis Report",
            "=" * 60,
            "",
            f"File: {audio_file}",
            f"Duration: {duration:.2f} seconds",
            f"Sample Rate: {sr} Hz",
            f"Total Samples: {len(audio_data):,}",
            "",
            "Mel Spectrogram:",
            f"  Shape: {mel_spec.shape}",
            f"  Min: {np.min(mel_spec):.4f}",
            f"  Max: {np.max(mel_spec):.4f}",
            f"  Mean: {np.mean(mel_spec):.4f}",
            f"  Std: {np.std(mel_spec):.4f}",
            "",
            "NMF Decomposition:",
            f"  Components: {pipeline_config['nmf']['n_components']}",
            f"  W (basis) shape: {W.shape}",
            f"  H (activations) shape: {H.shape}",
            f"  W min/max: [{np.min(W):.4f}, {np.max(W):.4f}]",
            f"  H min/max: [{np.min(H):.4f}, {np.max(H):.4f}]",
            "",
            "Extracted Features:",
            f"  Vector shape: {features.shape}",
            f"  Vector min: {np.min(features):.4f}",
            f"  Vector max: {np.max(features):.4f}",
            f"  Vector mean: {np.mean(features):.4f}",
            f"  Vector std: {np.std(features):.4f}",
        ]

        # Add classification result if model provided
        if model:
            click.echo(click.style("[*] Classifying with model...", fg="cyan"))
            pipeline = UnderWaterAudioPipeline(config=pipeline_config, verbose=verbose)
            pipeline.classifier = SoundClassifier.load(model)
            result = pipeline.classify(audio_file)

            report_lines.extend([
                "",
                "Classification Result:",
                f"  Predicted class: {result['predicted_label']}",
                f"  Confidence: {result['confidence']:.4f}",
                f"  Class probabilities: {result['probabilities']}",
            ])

        report_lines.append("=" * 60)

        # Save report
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        click.echo(f"\n    Report saved to: {click.style(report_path, fg='green')}")

        # Display summary
        click.echo(click.style("\n[✓] Analysis Summary:", fg="green"))
        for line in report_lines[3:]:  # Skip header
            if line:
                click.echo(line)

        click.echo(click.style("\n[✓] Analysis completed!", fg="green"))

    except Exception as e:
        click.echo(click.style(f"\n[✗] Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


# ============================================================================
# Config Command
# ============================================================================

@main.command()
@click.option(
    '--show',
    is_flag=True,
    help='Print default configuration.'
)
@click.option(
    '--generate',
    type=click.Path(),
    default=None,
    help='Generate config template to file.'
)
@click.option(
    '--validate',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    help='Validate config file.'
)
def config(
    show: bool,
    generate: Optional[str],
    validate: Optional[str]
) -> None:
    """
    Manage pipeline configuration.

    Examples:
        underwater-audio config --show
        underwater-audio config --generate ./my_config.yaml
        underwater-audio config --validate ./config.yaml
    """
    try:
        if not (show or generate or validate):
            click.echo(click.style("Error: Specify --show, --generate, or --validate", fg="red"), err=True)
            sys.exit(1)

        # Show default config
        if show:
            click.echo(click.style("[*] Default Configuration:", fg="cyan"))
            pipeline = UnderWaterAudioPipeline()
            config_yaml = yaml.dump(pipeline.config, default_flow_style=False)
            click.echo(config_yaml)

        # Generate config template
        if generate:
            click.echo(click.style("[*] Generating config template...", fg="cyan"))
            pipeline = UnderWaterAudioPipeline()
            os.makedirs(os.path.dirname(generate) or '.', exist_ok=True)
            save_pipeline_config(pipeline.config, generate)
            click.echo(f"    Config template saved to: {click.style(generate, fg='green')}")

        # Validate config
        if validate:
            click.echo(click.style("[*] Validating config file...", fg="cyan"))
            with open(validate, 'r') as f:
                config_data = yaml.safe_load(f)

            # Check required sections
            required_sections = ['preprocessor', 'nmf', 'classifier']
            missing = [s for s in required_sections if s not in config_data]

            if missing:
                click.echo(click.style(f"[✗] Missing sections: {', '.join(missing)}", fg="red"), err=True)
                sys.exit(1)

            # Check key parameters
            checks = [
                ('preprocessor', 'n_fft'),
                ('preprocessor', 'n_mels'),
                ('nmf', 'n_components'),
                ('classifier', 'model_type'),
            ]

            for section, key in checks:
                if key not in config_data[section]:
                    click.echo(click.style(f"[✗] Missing {section}.{key}", fg="red"), err=True)
                    sys.exit(1)

            click.echo(click.style("[✓] Config is valid!", fg="green"))

    except Exception as e:
        click.echo(click.style(f"[✗] Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

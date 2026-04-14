"""
CLI Implementation Documentation - Developer Guide

This document provides technical details about the CLI implementation,
architecture, and extension points for developers.
"""

# ============================================================================
# ARCHITECTURE OVERVIEW
# ============================================================================

## File Structure

src/cli.py (723 lines)
├── Imports (libraries and project modules)
├── Logging configuration
├── Helper functions (8 functions)
├── Main click group
├── Train command
├── Classify command
├── Batch classify command
├── Analyze command
├── Config command
└── Entry point

src/__main__.py (10 lines)
└── Entry point for: python -m src

tests/test_cli.py (370 lines)
├── Test helpers
├── Test main command
├── Test train command
├── Test classify command
├── Test batch-classify command
├── Test analyze command
└── Test config command


## Design Patterns

### 1. Command Group Pattern
Uses Click's @click.group() decorator to create a command group with subcommands:

    @click.group()
    def main():
        pass

    @main.command()
    def train(...):
        pass

Advantages:
- Professional command organization
- Automatic help generation
- Subcommand discovery

### 2. Helper Functions Pattern
Reusable functions for common operations:

    - load_pipeline_config(): Load YAML configuration
    - save_pipeline_config(): Save YAML configuration
    - validate_audio_dir(): Directory structure validation
    - get_label_mapping(): Class-to-label mapping
    - format_metrics(): Pretty-print metrics

Advantages:
- Reduced code duplication
- Easier testing
- Better maintainability

### 3. Error Handling Pattern
Consistent error handling across commands:

    try:
        # Command logic
    except Exception as e:
        click.echo(click.style(f"[✗] Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)

Advantages:
- User-friendly error messages
- Proper exit codes
- Colored output for visibility

### 4. Progress Indication Pattern
Visual feedback for long operations:

    with click.progressbar(items, label="Processing"):
        for item in items:
            # Process item

Advantages:
- User knows command is working
- Time estimation
- Cancellable operations


# ============================================================================
# COMMAND IMPLEMENTATIONS
# ============================================================================

## Train Command

Location: Lines 199-347
Purpose: Train ML classifier on labeled audio dataset

Key Functions:
1. validate_audio_dir() - Scan directory structure
2. get_label_mapping() - Create numeric labels
3. UnderWaterAudioPipeline.train() - Train pipeline
4. SoundClassifier.save() - Serialize model

Features:
- Progress bar for audio processing
- Stratified train/test split
- Multiple classifier types (RF, SVM, GB)
- Configurable NMF components
- Metrics display (accuracy, precision, recall, F1)
- Config auto-save

Flow:
1. Validate data directory structure
2. Enumerate audio files and create labels
3. Initialize pipeline with config
4. Process audio files (extract NMF features)
5. Create dataset and stratified split
6. Train classifier
7. Evaluate on train and test sets
8. Display metrics
9. Save model and config

## Classify Command

Location: Lines 354-447
Purpose: Classify single audio file with confidence

Key Functions:
1. find_config_file() - Auto-detect config
2. load_pipeline_config() - Load config
3. SoundClassifier.load() - Deserialize model
4. UnderWaterAudioPipeline.classify() - Classify

Features:
- Auto-detect config from model directory
- JSON export of results
- Verbose mode shows probabilities
- Confidence score

Flow:
1. Auto-detect or load config
2. Initialize pipeline
3. Load trained model
4. Classify audio
5. Display results
6. Optionally save JSON

## Batch Classify Command

Location: Lines 454-580
Purpose: Classify multiple audio files

Key Functions:
1. find_config_file() - Auto-detect config
2. SoundClassifier.load() - Deserialize model
3. UnderWaterAudioPipeline.classify() - Classify each file

Features:
- Progress bar for file processing
- Error handling per file
- CSV export
- Class distribution statistics
- Summary statistics

Flow:
1. Scan input directory for audio files
2. Auto-detect or load config
3. Load trained model
4. Process each file with progress bar
5. Collect results with error handling
6. Display summary and class distribution
7. Optionally export CSV

## Analyze Command

Location: Lines 587-711
Purpose: Generate diagnostic analysis of audio

Key Functions:
1. load_audio() - Load WAV file
2. AudioPreprocessor.preprocess() - Generate mel spectrogram
3. NMFFeatureExtractor.extract() - Extract features
4. SoundClassifier.load() - Load model (optional)
5. UnderWaterAudioPipeline.classify() - Classify (optional)

Features:
- Audio properties display
- Mel spectrogram statistics
- NMF decomposition details
- Feature vector analysis
- Optional classification result
- Text-based report generation

Output:
- analysis_report.txt with comprehensive details
- Console summary

## Config Command

Location: Lines 718-817
Purpose: Configuration management

Key Functions:
1. load_pipeline_config() - Load YAML
2. save_pipeline_config() - Save YAML
3. yaml validation - Check required sections

Features:
- Show default configuration
- Generate config template
- Validate config file

Flow:
1. Show default config (if --show)
2. Generate template (if --generate)
3. Validate config (if --validate)


# ============================================================================
# KEY HELPER FUNCTIONS
# ============================================================================

### load_pipeline_config(config_path: Optional[str]) -> Dict

Loads pipeline configuration from YAML file or returns default.

Parameters:
- config_path: Path to YAML config file

Returns:
- Dictionary with keys: preprocessor, nmf, classifier, label_names

Usage:
    config = load_pipeline_config('./config.yaml')

### save_pipeline_config(config: Dict, filepath: str) -> None

Saves configuration to YAML file with nice formatting.

Parameters:
- config: Configuration dictionary
- filepath: Output file path

Usage:
    save_pipeline_config(pipeline.config, './models/config.yaml')

### find_config_file(model_dir: str) -> Optional[str]

Auto-detects config.yaml in model directory.

Parameters:
- model_dir: Directory containing model file

Returns:
- Path to config.yaml if found, None otherwise

Usage:
    config_path = find_config_file(os.path.dirname('./models/classifier.pkl'))

### validate_audio_dir(data_dir: str) -> Dict[str, List[str]]

Scans directory for subdirectories with audio files.

Parameters:
- data_dir: Root directory to scan

Returns:
- Dictionary mapping class name to list of audio file paths

Raises:
- ValueError: If no subdirs or audio files found

Usage:
    class_files = validate_audio_dir('./training_data')
    # {'AUV': ['path/to/file1.wav', ...], 'Torpedo': [...]}

### get_label_mapping(class_names: List[str]) -> Dict[str, int]

Creates numeric labels from class names (alphabetically sorted).

Parameters:
- class_names: List of class names

Returns:
- Dictionary mapping class name to numeric ID

Usage:
    labels = get_label_mapping(['Torpedo', 'AUV', 'Vessel'])
    # {'AUV': 0, 'Torpedo': 1, 'Vessel': 2}

### format_metrics(metrics: Dict, decimals: int = 4) -> str

Formats metrics dictionary as human-readable string.

Parameters:
- metrics: Dictionary with metric names and values
- decimals: Decimal places for formatting

Returns:
- Formatted string with aligned output

Usage:
    metrics_str = format_metrics(train_metrics)
    click.echo(metrics_str)


# ============================================================================
# INTEGRATION WITH CORE MODULES
# ============================================================================

## Audio Loading
Pipeline: audio_loader.load_audio()
- Loads WAV files with librosa
- Returns (audio_data, sample_rate)
- Handles mono and stereo

## Preprocessing
Pipeline: preprocessor.AudioPreprocessor()
- Generates mel spectrograms
- Applies STFT, spectral gating, HPSS
- Configurable frequency bands

## Feature Extraction
Pipeline: feature_extractor.NMFFeatureExtractor()
- Decomposes mel spectrograms using NMF
- Extracts statistical features
- Returns normalized feature vectors

## Classification
Pipeline: classifier.SoundClassifier()
- Trains ML models (RF, SVM, GB)
- Makes predictions with probabilities
- Evaluates performance
- Serializes/deserializes models

## Configuration
Pipeline: pipeline.UnderWaterAudioPipeline()
- Orchestrates all components
- Manages configuration
- Provides unified interface


# ============================================================================
# ERROR HANDLING STRATEGY
# ============================================================================

## Validation Errors
- Checked at command argument parsing
- Non-existent directories/files caught by Click
- Custom validation in helper functions

## Processing Errors
- Individual files: Logged, counted separately (batch-classify)
- Critical failures: Exit with error message
- Graceful degradation where possible

## Error Messages
Format: [✗] Error: <description>
- User-friendly
- Actionable suggestions
- Colored red for visibility

## Exit Codes
- 0: Success
- 1: Processing/runtime error
- 2: Command-line usage error (handled by Click)


# ============================================================================
# TESTING STRATEGY
# ============================================================================

Test Coverage: tests/test_cli.py (370 lines)

## Test Categories

### Helper Functions (4 tests)
- Label mapping creation
- Metrics formatting
- Audio directory validation
- Error cases for validation

### Main Command (2 tests)
- Help output
- Version flag

### Train Command (2 tests)
- Help output
- Error handling for missing args

### Classify Command (2 tests)
- Help output
- Error handling

### Batch Classify Command (2 tests)
- Help output
- Error handling

### Analyze Command (2 tests)
- Help output
- Error handling

### Config Command (4 tests)
- Help output
- --show functionality
- --generate functionality
- Error handling

## Testing Approach

- Uses Click's CliRunner for command invocation
- Isolated testing of helper functions
- Mock file system with tmp_path
- Exit code validation
- Output text validation


# ============================================================================
# EXTENDING THE CLI
# ============================================================================

## Adding a New Command

Example: Add a "compare" command to compare models

```python
@main.command()
@click.option('--model1', required=True, type=click.Path(exists=True))
@click.option('--model2', required=True, type=click.Path(exists=True))
@click.option('--test-audio', required=True, type=click.Path(exists=True))
def compare(model1: str, model2: str, test_audio: str) -> None:
    """Compare predictions from two models on test audio."""
    try:
        click.echo(click.style("[*] Loading models...", fg="cyan"))
        clf1 = SoundClassifier.load(model1)
        clf2 = SoundClassifier.load(model2)
        
        click.echo(click.style("[*] Processing audio...", fg="cyan"))
        pipeline = UnderWaterAudioPipeline()
        audio_data, sr = load_audio(test_audio)
        mel_spec = pipeline.preprocessor.preprocess(audio_data, sr)
        features = pipeline.feature_extractor.extract(mel_spec)
        
        # Classify with both models
        pred1 = clf1.predict(features.reshape(1, -1))[0]
        prob1 = clf1.predict_proba(features.reshape(1, -1))[0]
        
        pred2 = clf2.predict(features.reshape(1, -1))[0]
        prob2 = clf2.predict_proba(features.reshape(1, -1))[0]
        
        click.echo(click.style("[✓] Results:", fg="green"))
        click.echo(f"  Model 1: {pred1} ({max(prob1):.4f})")
        click.echo(f"  Model 2: {pred2} ({max(prob2):.4f})")
        
    except Exception as e:
        click.echo(click.style(f"[✗] Error: {str(e)}", fg="red"), err=True)
        sys.exit(1)
```

## Adding a New Option to Existing Command

Example: Add pruning to train command

```python
@main.command()
@click.option('--data-dir', required=True, ...)
@click.option('--prune-threshold', type=float, default=0.01,
              help='Feature importance threshold for pruning')
def train(data_dir: str, ..., prune_threshold: float) -> None:
    """Train command with optional feature pruning."""
    # ... existing code ...
    
    if prune_threshold > 0:
        click.echo(f"Pruning features with importance < {prune_threshold}")
        # Implement feature pruning logic
```

## Customizing Output Format

Modify display functions:

```python
def format_metrics_table(metrics: Dict) -> str:
    """Format metrics as ASCII table."""
    import texttable
    table = texttable.Texttable()
    table.add_rows([['Metric', 'Value']] + list(metrics.items()))
    return table.draw()
```

## Adding Configuration Validation

Extend config validation:

```python
@config.command()
def validate(config_path: str):
    """Enhanced validation with detailed checks."""
    # ... load config ...
    
    # Validate model type
    if config_data['classifier']['model_type'] not in ['rf', 'svm', 'gb']:
        raise ValueError("Invalid model_type")
    
    # Validate frequency ranges
    if config_data['preprocessor']['min_freq'] >= config_data['preprocessor']['max_freq']:
        raise ValueError("min_freq must be less than max_freq")
```


# ============================================================================
# PERFORMANCE CONSIDERATIONS
# ============================================================================

## Memory Management
- Audio is loaded file-by-file (no all-in-memory storage)
- Progress bar avoids storing all filenames in memory
- Large datasets can be split into batches

## Computation Optimization
- sklearn NMF is faster than custom multiplicative update
- RandomForest parallelizes across cores (n_jobs=-1)
- Batch processing can use multiprocessing

## I/O Optimization
- YAML loading cached (single parse per config)
- Model serialization uses joblib (efficient binary format)
- CSV writing is streamed (no all-in-memory buffering)

## Bottlenecks
1. Audio loading: 50-100ms per file (network drive slower)
2. Preprocessing: 100-500ms per file (depends on duration)
3. Feature extraction: 50-200ms per file (NMF computation)
4. Classification: 1-10ms per file (ML inference fast)


# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================

## Logging Levels

Controlled by --verbose flag:

- INFO (default): High-level progress messages
- DEBUG (--verbose): Detailed processing information

## Debug Output

Enable with:
    python -m src train --data-dir ./data --verbose

## Troubleshooting Checklist

1. Check data directory structure
   - Subdirectories exist
   - Audio files present in subdirectories
   
2. Verify audio format
   - Supported: WAV, MP3, FLAC
   - Readable by librosa
   
3. Check model file
   - File exists and is readable
   - Is proper joblib serialized model
   - Config.yaml present in same directory
   
4. Check configuration
   - YAML syntax valid
   - Required sections present
   - Values in valid ranges


# ============================================================================
# FUTURE ENHANCEMENTS
# ============================================================================

Potential improvements:

1. Data Augmentation
   - Time stretching
   - Pitch shifting
   - Noise injection

2. Hyperparameter Tuning
   - Grid search
   - Random search
   - Bayesian optimization

3. Model Ensemble
   - Voting classifier
   - Stacking
   - Boosting

4. Cross-Validation
   - K-fold CV
   - Leave-one-out
   - Time series CV

5. Feature Visualization
   - t-SNE plots
   - Feature importance
   - Confusion matrices

6. API Server
   - Flask/FastAPI wrapper
   - RESTful endpoints
   - Real-time inference

7. Monitoring
   - Model performance tracking
   - Data drift detection
   - Prediction logging

8. Advanced Audio Processing
   - Beamforming
   - Matched filtering
   - Adaptive thresholding


# ============================================================================
# DEPENDENCIES & VERSIONS
# ============================================================================

Core Dependencies:
- click >= 7.1.0          (CLI framework)
- numpy >= 1.19.0         (Numerical computing)
- scikit-learn >= 0.24.0  (ML models, metrics)
- scipy >= 1.5.0          (Scientific computing, stats)
- librosa >= 0.8.0        (Audio processing)
- pyyaml >= 5.3.0         (Config file parsing)
- soundfile >= 0.10.0     (WAV file I/O)
- joblib >= 1.0.0         (Model serialization)

Development Dependencies:
- pytest >= 6.0.0         (Testing framework)
- pytest-cov >= 2.12.0    (Coverage reporting)

Tested Python Versions:
- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10


# ============================================================================
# CODE STYLE & CONVENTIONS
# ============================================================================

Followed Conventions:
- Type hints on all functions
- Comprehensive docstrings (Google style)
- PEP 8 code style
- Descriptive variable names
- Comments only when necessary
- No code duplication

Validation:
- All public functions have docstrings
- All parameters documented
- Return types specified
- Exceptions documented
- Examples provided where helpful
"""

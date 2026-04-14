"""
CLI User Guide - Underwater Vehicle Sound Classifier

This document provides comprehensive usage examples and documentation for the
command-line interface (CLI) of the underwater vehicle sound classifier.
"""

# ============================================================================
# INSTALLATION & SETUP
# ============================================================================

Installation and setup is straightforward. The CLI is implemented using Click,
a professional Python framework for command-line applications.

## Quick Start

1. Ensure dependencies are installed:
   pip install click numpy scikit-learn scipy librosa pyyaml soundfile

2. Run the CLI directly:
   python -m src

   Or use it as a module entry point if installed:
   underwater-audio --help

3. Verify installation:
   python -m src --version


# ============================================================================
# COMMAND REFERENCE
# ============================================================================

## 1. TRAIN COMMAND
Train a machine learning classifier on underwater audio data.

### Usage:
    underwater-audio train --data-dir <DIR> [OPTIONS]

### Required Arguments:
    --data-dir PATH              Directory with audio subdirectories (one per class)

### Optional Arguments:
    --model-type {rf,svm,gb}     Model type (rf=RandomForest, svm=SVM, gb=GradientBoosting)
                                  Default: rf
    --output-model PATH          Path to save trained model
                                  Default: ./models/classifier.pkl
    --test-size FLOAT            Train/test split ratio (0.0-1.0)
                                  Default: 0.2
    --n-components INT           Number of NMF components
                                  Default: 6
    --verbose                    Enable verbose logging

### Example:
    underwater-audio train \
      --data-dir ./training_data \
      --model-type rf \
      --output-model ./models/classifier.pkl \
      --test-size 0.2 \
      --n-components 6 \
      --verbose

### Data Directory Structure:
    training_data/
    ├── AUV/
    │   ├── sample1.wav
    │   ├── sample2.wav
    │   └── ...
    ├── Torpedo/
    │   ├── sample1.wav
    │   ├── sample2.wav
    │   └── ...
    └── Surface_Vessel/
        ├── sample1.wav
        └── ...

### Output:
    - Trained model saved to specified path (default: ./models/classifier.pkl)
    - Configuration saved alongside model (config.yaml)
    - Training metrics displayed:
      * Training accuracy, precision, recall, F1
      * Test accuracy, precision, recall, F1

### Model Types:
    - rf (RandomForest):         Ensemble model, robust to overfitting
    - svm (SVM):                 Good generalization, high-dimensional
    - gb (GradientBoosting):     Sequential boosting, often highest accuracy


## 2. CLASSIFY COMMAND
Classify a single audio file using a trained model.

### Usage:
    underwater-audio classify --audio-file <FILE> --model <MODEL> [OPTIONS]

### Required Arguments:
    --audio-file PATH            Path to WAV file to classify
    --model PATH                 Path to trained classifier model

### Optional Arguments:
    --config PATH                Path to config YAML (auto-detected from model dir)
    --save-json PATH             Save results to JSON file
    --verbose                    Show detailed output (probabilities)

### Example:
    underwater-audio classify \
      --audio-file ./samples/auv_sample.wav \
      --model ./models/classifier.pkl \
      --verbose \
      --save-json results.json

### Output:
    - Predicted class
    - Confidence score (0-1)
    - Class probabilities (if --verbose)
    - Optional JSON file with complete results

### JSON Output Format:
    {
      "audio_file": "./samples/auv_sample.wav",
      "predicted_class": 0,
      "predicted_label": "AUV",
      "confidence": 0.95,
      "probabilities": [0.95, 0.04, 0.01]
    }


## 3. BATCH-CLASSIFY COMMAND
Classify multiple audio files in a directory.

### Usage:
    underwater-audio batch-classify --input-dir <DIR> --model <MODEL> [OPTIONS]

### Required Arguments:
    --input-dir PATH             Directory containing WAV files to classify
    --model PATH                 Path to trained classifier model

### Optional Arguments:
    --config PATH                Path to config YAML (auto-detected)
    --output-csv PATH            Save results to CSV file
    --verbose                    Show progress details

### Example:
    underwater-audio batch-classify \
      --input-dir ./test_audio \
      --model ./models/classifier.pkl \
      --output-csv results.csv \
      --verbose

### Output:
    - Summary statistics (total files, successful, failed)
    - Class distribution with percentages
    - Optional CSV file with detailed results

### CSV Output Format:
    filename,filepath,predicted_class,predicted_label,confidence
    sample1.wav,./test_audio/sample1.wav,0,AUV,0.95
    sample2.wav,./test_audio/sample2.wav,1,Torpedo,0.87
    sample3.wav,./test_audio/sample3.wav,2,Surface_Vessel,0.92

### Features:
    - Progress bar showing processing status
    - Error handling for individual files
    - Class distribution histogram
    - CSV export for downstream analysis


## 4. ANALYZE COMMAND
Analyze audio files and generate diagnostic plots.

### Usage:
    underwater-audio analyze --audio-file <FILE> [OPTIONS]

### Required Arguments:
    --audio-file PATH            Path to WAV file to analyze

### Optional Arguments:
    --output-dir PATH            Directory to save plots
                                  Default: ./diagnostics
    --model PATH                 Trained model for classification (optional)
    --config PATH                Path to config YAML
    --verbose                    Show detailed output

### Example:
    underwater-audio analyze \
      --audio-file ./samples/auv_sample.wav \
      --output-dir ./analysis \
      --model ./models/classifier.pkl \
      --verbose

### Output:
    - Analysis report (analysis_report.txt):
      * Audio duration, sample rate, total samples
      * Mel spectrogram statistics
      * NMF decomposition details
      * Classification result (if model provided)
    - Console summary of audio properties
    - Feature statistics

### Analysis Report Contents:
    ====================================================
    Audio Analysis Report
    ====================================================

    File: ./samples/auv_sample.wav
    Duration: 10.25 seconds
    Sample Rate: 16000 Hz
    Total Samples: 164,000

    Mel Spectrogram:
      Shape: (512, 647)
      Min: 0.0001
      Max: 0.8934
      Mean: 0.1234
      Std: 0.2156

    NMF Decomposition:
      Components: 6
      W (basis) shape: (512, 6)
      H (activations) shape: (6, 647)
      W min/max: [0.0, 1.0]
      H min/max: [0.0, 5.2]

    Extracted Features:
      Vector shape: (48,)
      Vector min: 0.0000
      Vector max: 1.0000
      Vector mean: 0.5123
      Vector std: 0.2456

    Classification Result:
      Predicted class: AUV
      Confidence: 0.94
      Class probabilities: [0.94, 0.05, 0.01]


## 5. CONFIG COMMAND
Manage pipeline configuration.

### Usage:
    underwater-audio config [OPTIONS]

### Optional Arguments:
    --show                       Display default configuration
    --generate PATH              Generate config template to file
    --validate PATH              Validate existing config file

### Examples:

    # Show default configuration
    underwater-audio config --show

    # Generate new config file
    underwater-audio config --generate ./my_config.yaml

    # Validate existing config
    underwater-audio config --validate ./config.yaml

### Configuration Structure:
    preprocessor:
      n_fft: 8192
      min_freq: 200
      max_freq: 12000
      n_mels: 512
      noise_gate_multiplier: 1.5
      hpss_margin: 3.0

    nmf:
      n_components: 6
      use_sklearn: true
      max_iter: 500

    classifier:
      model_type: 'rf'
      random_state: 42

    label_names:
      '0': 'AUV'
      '1': 'Torpedo'
      '2': 'Surface_Vessel'

### Configuration Parameters:

    Preprocessor:
      - n_fft: FFT window size for STFT
      - min_freq: Minimum frequency to analyze (Hz)
      - max_freq: Maximum frequency to analyze (Hz)
      - n_mels: Number of Mel frequency bands
      - noise_gate_multiplier: Spectral gating strength
      - hpss_margin: Harmonic-percussive separation

    NMF:
      - n_components: Number of basis functions
      - use_sklearn: Use sklearn vs custom NMF
      - max_iter: Maximum iterations for convergence

    Classifier:
      - model_type: ML algorithm (rf/svm/gb)
      - random_state: Random seed for reproducibility

    Label Names:
      - Mapping of class IDs to human-readable names


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

This section demonstrates a complete workflow from data preparation to
batch classification.

## Step 1: Prepare Training Data
Create a directory structure with training audio files organized by class:

    mkdir -p training_data/{AUV,Torpedo,Surface_Vessel}
    cp auv_samples/*.wav training_data/AUV/
    cp torpedo_samples/*.wav training_data/Torpedo/
    cp vessel_samples/*.wav training_data/Surface_Vessel/

## Step 2: Train Model
Train a RandomForest classifier:

    underwater-audio train \
      --data-dir training_data \
      --model-type rf \
      --output-model ./models/classifier.pkl \
      --n-components 6 \
      --verbose

    Output:
    [*] Scanning data directory...
        Found 3 classes: AUV, Surface_Vessel, Torpedo
        Total audio files: 150
    [*] Initializing pipeline...
    [*] Training model...
        Processing audio files |████████...| 100%
    [*] Evaluating model...

    [✓] Training Metrics:
      accuracy.........  0.9200
      precision.........  0.9150
      recall............  0.9180
      f1...............  0.9165

    [✓] Test Metrics:
      accuracy.........  0.8800
      precision.........  0.8750
      recall............  0.8820
      f1...............  0.8785

    [✓] Training completed successfully!

## Step 3: Classify Single Audio File
Test the model on a single file:

    underwater-audio classify \
      --audio-file test_samples/unknown_01.wav \
      --model models/classifier.pkl \
      --verbose

    Output:
    [✓] Classification Results:
      Audio file:................... test_samples/unknown_01.wav
      Predicted class:.............. AUV
      Confidence:................... 0.9234

    [*] Class Probabilities:
        Class 0: 0.9234
        Class 1: 0.0512
        Class 2: 0.0254

## Step 4: Batch Classify Directory
Classify all test files:

    underwater-audio batch-classify \
      --input-dir test_samples \
      --model models/classifier.pkl \
      --output-csv results.csv

    Output:
    [✓] Batch Classification Results:
      Total files:..................  50
      Successfully classified:......  49
      Failed:........................  1

    [*] Class Distribution:
        AUV........................  20 ( 40.8%)
        Surface_Vessel............. 18 ( 36.7%)
        Torpedo....................  11 ( 22.4%)

        Results saved to: results.csv

## Step 5: Analyze Specific File
Generate diagnostic analysis:

    underwater-audio analyze \
      --audio-file test_samples/auv_01.wav \
      --output-dir diagnostics \
      --model models/classifier.pkl

    Output:
    [*] Loading audio file...
      Duration:................... 8.45 seconds
      Sample rate:................ 16000 Hz
      Samples:.................... 135200
    [*] Preprocessing audio...
        Mel spectrogram shape: (512, 535)
    [*] Extracting NMF features...
        NMF basis shape: (512, 6)
        NMF activations shape: (6, 535)
        Feature vector shape: (48,)
    [*] Classifying with model...

    [✓] Analysis Summary:
    ====================================================
    Audio Analysis Report
    ...


# ============================================================================
# ERROR HANDLING & TROUBLESHOOTING
# ============================================================================

## Common Errors:

### Missing Data Directory
Error: Data directory does not exist: ./data
Solution: Create the directory or check the path

### No Audio Files Found
Error: No audio files found in subdirectories of ./data
Solution: Ensure audio files (*.wav, *.mp3, *.flac) are in subdirectories

### Model File Not Found
Error: Path is not a file: ./models/classifier.pkl
Solution: Train a model first or check the file path

### Invalid Audio File
Error: Failed to read audio file: ...
Solution: Ensure the file is a valid WAV/MP3/FLAC format

### Out of Memory
For large datasets, process in batches or use a machine with more RAM.


# ============================================================================
# LOGGING & DEBUG MODE
# ============================================================================

Use the --verbose flag to enable detailed logging:

    underwater-audio train --data-dir ./data --verbose
    underwater-audio classify --audio-file test.wav --model classifier.pkl --verbose

Verbose mode outputs:
- Detailed processing steps
- Individual file processing information
- Feature extraction statistics
- Model training details


# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

1. NMF Components:
   - More components (6-12) capture finer details but increase computation
   - Fewer components (3-6) reduce computation time
   - Balance based on available computational resources

2. Model Selection:
   - RandomForest (rf): Good baseline, fast training
   - SVM (svm): Better accuracy on small datasets
   - GradientBoosting (gb): Often best accuracy, slower training

3. Batch Processing:
   - Use --output-csv to save results for analysis
   - Process large directories in batches with separate input folders
   - Use --verbose to monitor progress

4. Audio Preprocessing:
   - Default settings work for most underwater recordings
   - Adjust n_mels for different frequency resolution
   - Adjust noise_gate_multiplier for heavy noise conditions


# ============================================================================
# ADVANCED USAGE
# ============================================================================

### Custom Configuration
Create a custom config file and use it:

    underwater-audio config --generate my_config.yaml
    # Edit my_config.yaml with custom parameters
    underwater-audio classify --audio-file test.wav --model classifier.pkl --config my_config.yaml

### Validate Configuration
Check if a config file is valid:

    underwater-audio config --validate my_config.yaml

### Auto-Detection
The CLI automatically detects config.yaml in the model directory:

    # If ./models/config.yaml exists, it's auto-loaded
    underwater-audio classify --audio-file test.wav --model ./models/classifier.pkl

### JSON Export
Save classification results for programmatic access:

    underwater-audio classify --audio-file test.wav --model classifier.pkl --save-json result.json

    # result.json contains all probabilities and metadata
    cat result.json


# ============================================================================
# VERSION & HELP
# ============================================================================

View version information:
    underwater-audio --version

View help for main command:
    underwater-audio --help

View help for specific command:
    underwater-audio train --help
    underwater-audio classify --help
    underwater-audio batch-classify --help
    underwater-audio analyze --help
    underwater-audio config --help


# ============================================================================
# SYSTEM REQUIREMENTS
# ============================================================================

Python:
    - Python 3.7 or later

Dependencies:
    - click >= 7.1.0
    - numpy >= 1.19.0
    - scikit-learn >= 0.24.0
    - scipy >= 1.5.0
    - librosa >= 0.8.0
    - pyyaml >= 5.3.0
    - soundfile >= 0.10.0
    - joblib >= 1.0.0

Memory:
    - Minimum: 2 GB RAM
    - Recommended: 4 GB RAM for batch processing

Disk Space:
    - Models: ~10-50 MB per trained model
    - Diagnostics: ~1-5 MB per analyzed file


# ============================================================================
# EXIT CODES
# ============================================================================

0       Success
1       Error (file not found, invalid input, processing failure)
2       Command line usage error (missing required args, invalid options)


# ============================================================================
# CONTACT & SUPPORT
# ============================================================================

For issues or questions about the CLI, refer to:
- Test suite documentation in tests/README.md
- Source code documentation in src/cli.py
- Configuration guide in config.yaml
"""

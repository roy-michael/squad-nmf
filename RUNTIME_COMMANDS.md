# Runtime Commands - How to Run the System

## Main Runtime Command

The primary way to run the underwater audio classifier:

```bash
python -m src
```

This is the module entry point. Add subcommands after it.

---

## Available Subcommands

### 1. **CLASSIFY** - Classify a single audio file

```bash
python -m src classify --input audio.wav --model trained_model.pkl
```

**Parameters:**
- `--input FILE` - Path to WAV audio file to classify
- `--model MODEL` - Path to trained model file (pkl)
- `--verbose` - Show detailed logging (optional)

**Output:** Predicted class and confidence scores printed to console

**Example:**
```bash
python -m src classify --input underwater_recording.wav --model model.pkl
```

---

### 2. **BATCH-CLASSIFY** - Classify multiple audio files (RECOMMENDED)

```bash
python -m src batch-classify --input data/ --model model.pkl --output results.csv --n-jobs=-1
```

**Parameters:**
- `--input DIRECTORY` - Path to folder containing WAV files
- `--model MODEL` - Path to trained model
- `--output FILE` - CSV file to save results
- `--n-jobs N` - Number of workers (1, 2, 4, -1 for all cores)
- `--verbose` - Show detailed logging (optional)

**Output:** CSV file with predictions for each file

**Example (Parallel, 4 workers):**
```bash
python -m src batch-classify --input recordings/ --model model.pkl --n-jobs 4 --output results.csv
```

**Example (All cores):**
```bash
python -m src batch-classify --input recordings/ --model model.pkl --n-jobs=-1 --output results.csv
```

**Example (Single worker, low memory):**
```bash
python -m src batch-classify --input recordings/ --model model.pkl --n-jobs 1 --output results.csv
```

---

### 3. **TRAIN** - Train a new classification model

```bash
python -m src train --data training_data/ --output model.pkl --n-jobs=-1
```

**Data Structure Required:**
```
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
    ├── sample2.wav
    └── ...
```

**Parameters:**
- `--data DIRECTORY` - Path to training data folder (with class subdirectories)
- `--output MODEL` - Path to save trained model
- `--n-jobs N` - Number of workers (1, 2, 4, -1)
- `--verbose` - Show detailed logging (optional)

**Output:** Trained model file (pkl) saved to disk

**Example:**
```bash
python -m src train --data training_data/ --output model.pkl --n-jobs=-1
```

---

### 4. **CLUSTER** - Unsupervised K-Means clustering

```bash
python -m src cluster --input data/ --n-clusters 3 --output clusters.json
```

**Parameters:**
- `--input DIRECTORY` - Path to folder with WAV files
- `--n-clusters N` - Number of clusters to find
- `--output FILE` - JSON file to save cluster assignments
- `--verbose` - Show detailed logging (optional)

**Output:** JSON file mapping files to cluster IDs

**Example:**
```bash
python -m src cluster --input recordings/ --n-clusters 3 --output clusters.json
```

---

### 5. **TEMPORAL-CLUSTER** - Find multiple sound patterns within files

```bash
python -m src temporal-cluster --input data/ --n-patterns 3 --n-jobs=-1 --visualize --viz-dir plots/
```

**Parameters:**
- `--input DIRECTORY` - Path to folder with WAV files
- `--n-patterns N` - Number of temporal patterns to find
- `--n-jobs N` - Number of workers (1, 2, 4, -1)
- `--visualize` - Generate visualization plots
- `--viz-dir DIRECTORY` - Where to save visualization PNGs
- `--verbose` - Show detailed logging (optional)

**Output:** 
- JSON file with temporal clustering results
- PNG visualization plots (if --visualize enabled)

**Example (with visualizations):**
```bash
python -m src temporal-cluster --input recordings/ --n-patterns 3 --n-jobs=-1 --visualize --viz-dir ./plots/
```

**Example (without visualizations):**
```bash
python -m src temporal-cluster --input recordings/ --n-patterns 3 --n-jobs=-1
```

---

### 6. **ANALYZE** - Analyze audio features

```bash
python -m src analyze --input audio.wav --output analysis.json
```

**Parameters:**
- `--input FILE` - Path to WAV file
- `--output FILE` - JSON file to save analysis
- `--verbose` - Show detailed logging (optional)

**Output:** JSON file with extracted features

---

### 7. **CONFIG** - Manage configuration

```bash
python -m src config --show
```

**Subcommands:**
- `config --show` - Display current configuration
- `config --reset` - Reset to defaults

---

## Quick Reference Table

| Task | Command |
|------|---------|
| **Classify 1 file** | `python -m src classify --input file.wav --model model.pkl` |
| **Classify many files** | `python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1 --output results.csv` |
| **Train new model** | `python -m src train --data training_data/ --output model.pkl --n-jobs=-1` |
| **Find patterns in files** | `python -m src temporal-cluster --input data/ --n-patterns 3 --visualize --viz-dir plots/` |
| **Unsupervised clustering** | `python -m src cluster --input data/ --n-clusters 3 --output clusters.json` |
| **Analyze single file** | `python -m src analyze --input audio.wav --output analysis.json` |

---

## Real-World Examples

### Example 1: Complete Workflow from Scratch

```bash
# Step 1: Train model on labeled data
python -m src train --data training_data/ --output my_model.pkl --n-jobs=-1

# Step 2: Classify new recordings
python -m src batch-classify --input new_recordings/ --model my_model.pkl --n-jobs=-1 --output results.csv

# Step 3: Analyze temporal patterns
python -m src temporal-cluster --input new_recordings/ --n-patterns 3 --n-jobs=-1 --visualize --viz-dir ./analysis/
```

### Example 2: Quick Test (Low Resources)

```bash
# Train with limited resources
python -m src train --data training_data/ --output model.pkl --n-jobs 1

# Test on few files
python -m src batch-classify --input test_data/ --model model.pkl --n-jobs 1 --output test_results.csv
```

### Example 3: Production Batch Processing

```bash
# Classify large dataset efficiently
python -m src batch-classify \
  --input /mnt/data/recordings/ \
  --model production_model.pkl \
  --n-jobs=-1 \
  --output production_results.csv \
  --verbose
```

---

## Command Structure Breakdown

All commands follow this pattern:

```
python -m src COMMAND --param1 value1 --param2 value2
         │      │       │       │      │       │
         │      │       │       │      │       └─ Parameter values
         │      │       │       │      └─────── Parameter names
         │      │       │       └──────────── Parameter
         │      │       └─────────────────── More parameters
         │      └──────────────────────────── Command
         └─────────────────────────────────── Module entry point
```

---

## Getting Help

View all available commands:
```bash
python -m src --help
```

Get help for specific command:
```bash
python -m src classify --help
python -m src batch-classify --help
python -m src train --help
python -m src temporal-cluster --help
python -m src cluster --help
```

---

## Most Common Commands (Copy-Paste Ready)

### Command 1: Classify Multiple Files (MOST USED)
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs 4 --output results.csv
```

### Command 2: Train Model
```bash
python -m src train --data training_data/ --output model.pkl --n-jobs 4
```

### Command 3: Analyze Patterns with Visuals
```bash
python -m src temporal-cluster --input data/ --n-patterns 3 --n-jobs 4 --visualize --viz-dir plots/
```

### Command 4: Single File Classification
```bash
python -m src classify --input recording.wav --model model.pkl
```

---

## Environment Setup (Before Running)

### Install Dependencies
```bash
pip install click numpy scikit-learn scipy librosa pyyaml soundfile joblib
```

### Verify Installation
```bash
python -m src --help
```

Should show available commands without errors.

---

## File Locations

For commands to work, ensure:

1. **Model file** exists: `path/to/trained_model.pkl`
2. **Input audio** exists: `path/to/audio.wav` or `path/to/folder/`
3. **Output directory** is writable: `path/to/output/`
4. **Training data** structure is correct (class subdirectories)

---

## Return Values

### Exit Codes
- `0` - Success
- `1` - Runtime error (invalid input, missing files, processing error)
- `2` - Usage error (wrong arguments, missing required parameters)

### Output Files
Commands generate files based on `--output` parameter:
- **CSV files** - Plain text, one line per result
- **JSON files** - Structured data, readable in any text editor
- **PNG files** - Visualization plots in specified directory
- **PKL files** - Binary model files (use with `--model` parameter)

---

## Debugging Tips

Add `--verbose` to any command for detailed logging:
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs 4 --output results.csv --verbose
```

This shows:
- File processing progress
- Timing information
- Error details
- Feature extraction steps

---

## Performance Hints

- Use `--n-jobs=-1` for maximum speed (all CPU cores)
- Use `--n-jobs 1` if running into memory issues
- Adjust based on available RAM (see GPU_AND_PERFORMANCE_STATUS.md)
- Larger `n-jobs` = faster but uses more memory

---

## Summary

**The main entry point is always:**
```bash
python -m src COMMAND [OPTIONS]
```

**Most common usage:**
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs 4 --output results.csv
```

That's your runtime command! Adjust parameters as needed for your use case.


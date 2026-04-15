# Two-Phase Clustering & Classification Workflow

## Overview

Your underwater sound classifier now supports **two complementary approaches**:

1. **Unsupervised Discovery** (Phase 1): Use K-means to discover sound patterns in unlabeled recordings
2. **Supervised Classification** (Phase 2): Train a classifier on manually-labeled clusters

This document guides you through both phases.

---

## Phase 1: Unsupervised Sound Discovery

### Why K-Means?

Your recording dates (Location_2307, Location_2407_1, Location_2507_1) indicate **unlabeled data**. K-means helps you:

- Discover what sound types naturally exist in your ocean recordings
- Cluster similar recordings together automatically
- Prepare data for Phase 2 labeling

### Running Clustering

#### Basic Usage

```bash
cd D:\dev\squad-nmf

# Cluster your recordings into 3 natural groups
python -m src cluster \
  --input-dir ./data/Location_2307 \
  --n-clusters 3 \
  --plot-dir ./analysis_plots \
  --output-csv cluster_assignments.csv
```

#### Find Optimal Cluster Count

If you're unsure how many clusters exist:

```bash
python -m src cluster \
  --input-dir ./data/Location_2307 \
  --find-optimal \
  --plot-dir ./analysis_plots \
  --output-csv cluster_assignments.csv
```

This tests k=2 to k=10 and recommends the best cluster count using **silhouette scoring**.

#### Advanced: All Recordings at Once

```bash
# Combine all locations for discovery
cat data/Location_*/RBW*.wav > all_recordings.wav  # Or organize differently

python -m src cluster \
  --input-dir ./combined_recordings \
  --n-clusters 5 \
  --find-optimal \
  --verbose
```

### Understanding the Output

#### Cluster Assignments CSV

File: `cluster_assignments.csv`

```csv
filename,cluster
RBW6737_20250723_094600.wav,0
RBW6737_20250723_094700.wav,1
RBW6737_20250723_094800.wav,0
RBW6737_20250723_094900.wav,2
RBW6737_20250723_095000.wav,1
```

Cluster IDs (0, 1, 2, ...) indicate groupings. Identical IDs = similar acoustic properties.

#### Visualizations

Three plots are automatically generated:

1. **clusters_2d.png** - 2D scatter plot
   - X/Y axes: PCA components (dimensionality reduction)
   - Colors: Cluster assignments
   - Points: Individual recordings
   - **What to look for**: Are clusters visually separated?

2. **silhouette_analysis.png** - Cluster quality metrics
   - Horizontal bars: Per-sample silhouette coefficients
   - Each color: One cluster
   - Red line: Overall silhouette score (-1 to +1)
   - **Interpretation**:
     - Bars extending right = good samples (cohesive clusters)
     - Bars extending left = poor samples (ambiguous membership)
     - Red line > 0.5 = good separation
     - Red line < 0 = clusters overlap too much

3. **elbow_curve.png** (only with --find-optimal)
   - X-axis: Number of clusters (k)
   - Y-axis: Silhouette score
   - Red star: Best k value
   - **Usage**: Look for the "elbow" where improvement plateaus

#### Console Output Example

```
[*] Clustering ./data/Location_2307 into 3 clusters...

Clustering Results:
  Silhouette Score: +0.452
  Davies-Bouldin Index: 1.234
  Total Inertia: 4521.6

  Cluster 0: 2 files
    - RBW6737_20250723_094600.wav
    - RBW6737_20250723_094700.wav
    ... and 0 more

  Cluster 1: 2 files
    - RBW6737_20250723_094800.wav
    - RBW6737_20250723_094900.wav
    ... and 0 more

  Cluster 2: 1 files
    - RBW6737_20250723_095000.wav

✓ Saved cluster assignments: cluster_assignments.csv
✓ Saved scatter plot: ./analysis_plots/clusters_2d.png
✓ Saved silhouette plot: ./analysis_plots/silhouette_analysis.png

Next Steps:
1. Review clusters in: cluster_assignments.csv
2. Check visualizations in: ./analysis_plots/
3. Manually label clusters by sound type (vehicle type, etc.)
4. Create subdirectories with labeled data
5. Train supervised model: underwater-audio train --data-dir <labeled_data>
```

### Interpreting Clustering Quality

#### Metrics Explained

| Metric | Range | Better Is | Interpretation |
|--------|-------|-----------|-----------------|
| **Silhouette Score** | -1 to +1 | Higher | How distinct clusters are (-1=bad, 0=overlapping, +1=perfect) |
| **Davies-Bouldin Index** | 0 to ∞ | Lower | Cluster compactness (lower=tighter clusters) |
| **Inertia** | 0 to ∞ | Lower | Sum of squared distances to cluster centers |

#### Good Clustering Indicators

✅ Silhouette score > 0.5  
✅ Davies-Bouldin < 1.5  
✅ 2D scatter plot shows visually separated clouds  
✅ Silhouette plot has few negative bars  

#### Poor Clustering Indicators

❌ Silhouette score < 0 (clusters overlap)  
❌ Davies-Bouldin > 2 (loose clusters)  
❌ 2D plot is one blob  
❌ Silhouette plot has many negative bars  

### Handling Bad Clustering

If clustering looks poor:

1. **Increase/decrease k**
   - Try different cluster counts
   - Use `--find-optimal` to scan the range
   - K-means may not be the right method for your data

2. **Inspect your audio**
   - Are all recordings same quality/duration?
   - Do they contain distinct sound types?
   - May need preprocessing/filtering first

3. **Consider the features**
   - Current: 48-dim NMF features (6 components × 8 moments)
   - Adjust `--n-components` to capture different patterns
   - More components = more detailed (but noisier)
   - Fewer components = more abstract (but may lose detail)

---

## Phase 2: Manual Labeling & Preparation

### Step 1: Review Clusters

Open `cluster_assignments.csv` in Excel or a text editor:

```csv
filename,cluster
RBW6737_20250723_094600.wav,0
RBW6737_20250723_094700.wav,1
RBW6737_20250723_094800.wav,0
```

Listen to recordings in each cluster. Note patterns:

- **Cluster 0**: Low-frequency humming, consistent amplitude
- **Cluster 1**: High-frequency chirps, variable amplitude
- **Cluster 2**: Noisy, possibly background

### Step 2: Assign Labels

Based on your acoustic expertise, label each cluster:

| Cluster | Acoustic Pattern | Label | Vehicle Type (Guess) |
|---------|-----------------|-------|----------------------|
| 0 | Low-freq, steady | `AUV_steady` | Autonomous Vehicle |
| 1 | High-freq, chirping | `Torpedo_pulse` | Torpedo/Projectile |
| 2 | Noise | `Unknown` | Background/Noise |

### Step 3: Organize Training Data

Create a directory structure matching the supervised format:

```
labeled_data/
├── AUV_steady/
│   ├── RBW6737_20250723_094600.wav
│   ├── RBW6737_20250723_094700.wav
│   └── ...
├── Torpedo_pulse/
│   ├── RBW6737_20250723_094800.wav
│   └── ...
└── Unknown/
    └── RBW6737_20250723_095000.wav
```

**Important**: Subdirectory names become your class labels. Use simple names without spaces.

### Step 4: Move Files (Manual or Scripted)

**Option A: Manual in File Explorer**

Copy/move files into subdirectories based on cluster assignments.

**Option B: Python Script**

```python
import pandas as pd
import shutil
from pathlib import Path

# Load cluster assignments
clusters_df = pd.read_csv('cluster_assignments.csv')

# Define cluster-to-label mapping
label_map = {
    0: 'AUV_steady',
    1: 'Torpedo_pulse',
    2: 'Unknown'
}

# Create label directories
for label in label_map.values():
    Path(f'labeled_data/{label}').mkdir(parents=True, exist_ok=True)

# Move files
for _, row in clusters_df.iterrows():
    filename = row['filename']
    cluster = row['cluster']
    label = label_map[cluster]
    
    src = Path('data/Location_2307') / filename
    dst = Path(f'labeled_data/{label}/{filename}')
    
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"Moved {filename} to {label}/")
```

### Step 5: Verify Structure

```bash
# Check your labeled data
ls -la labeled_data/
```

Expected output:

```
labeled_data/
├── AUV_steady/          (2 files)
├── Torpedo_pulse/       (2 files)
└── Unknown/             (1 file)
```

---

## Phase 2b: Supervised Training

Once you have labeled data, train a classifier:

### Basic Training

```bash
python -m src train \
  --data-dir ./labeled_data \
  --model-type rf \
  --output-model ./models/classifier.pkl \
  --verbose
```

### With Custom Parameters

```bash
python -m src train \
  --data-dir ./labeled_data \
  --model-type rf \
  --output-model ./models/classifier.pkl \
  --test-size 0.2 \
  --n-components 8 \
  --verbose
```

### Expected Output

```
[*] Scanning data directory...
    Found 3 classes: AUV_steady, Torpedo_pulse, Unknown
    Total audio files: 5

[*] Initializing pipeline...
[*] Extracting NMF features...
[*] Training RandomForest classifier...

Training Results:
  Train Accuracy: 1.000
  Test Accuracy:  1.000
  
  Precision (per class):
    AUV_steady:     1.000
    Torpedo_pulse:  1.000
    Unknown:        1.000
  
  ...

✓ Model saved: ./models/classifier.pkl
✓ Config saved: ./models/config.yaml
```

---

## Integrated Workflow Example

Here's a complete end-to-end example:

### Step 1: Discover (30 minutes)

```bash
# Cluster all recordings from Location_2307
python -m src cluster \
  --input-dir ./data/Location_2307 \
  --find-optimal \
  --plot-dir ./step1_discovery \
  --output-csv ./step1_discovery/clusters.csv \
  --verbose
```

### Step 2: Inspect (15 minutes)

- Listen to samples from each cluster
- Review `step1_discovery/silhouette_analysis.png`
- Decide on labels based on acoustic patterns

### Step 3: Label Manually (20 minutes)

Create `labeled_data/` directory with subdirectories:
- `labeled_data/Vehicle_Type_A/` (files from cluster 0)
- `labeled_data/Vehicle_Type_B/` (files from cluster 1)
- etc.

### Step 4: Train Supervised (5 minutes)

```bash
python -m src train \
  --data-dir ./labeled_data \
  --model-type rf \
  --output-model ./models/classifier.pkl
```

### Step 5: Deploy (instant)

Use your trained model:

```bash
# Classify a new recording
python -m src classify \
  --audio-file ./new_recording.wav \
  --model ./models/classifier.pkl

# Classify many recordings
python -m src batch-classify \
  --input-dir ./test_recordings \
  --model ./models/classifier.pkl \
  --output-csv predictions.csv
```

**Total time: ~70 minutes** for end-to-end pipeline with 15 recordings

---

## Why This Two-Phase Approach?

### Advantages

✅ **No manual labeling needed for discovery** - Understand your data first  
✅ **Data-driven labels** - Let clustering reveal patterns before assigning names  
✅ **Quality assurance** - Review visualizations before training  
✅ **Flexibility** - Different labelings possible (by vehicle type, location, time, etc.)  
✅ **Reproducibility** - Clustering metrics document data quality  

### Compared to Supervised-Only

Traditional workflow: **Label 100 files manually → Train model**  
Your approach: **Cluster 100 files automatically → Label intelligently → Train model**

Result: **Faster, more accurate labeling** because you understand patterns first

---

## Troubleshooting

### Problem: "No WAV files found"

```
ValueError: No WAV files found in ./data/Location_2307
```

**Solution**: Check that:
- Directory exists and is readable
- Files have `.wav` extension (lowercase)
- Directory is flat (no subdirectories)

### Problem: "Clusters overlap (negative silhouette)"

```
Silhouette Score: -0.123
```

**Solution**: 
1. Try different k values (`--n-clusters 2`, `--n-clusters 5`, etc.)
2. Use `--find-optimal` to scan automatically
3. Inspect raw audio - may be too similar for separation

### Problem: "Feature dimension mismatch"

```
ValueError: X has 32 features, but model expects 48
```

**Solution**: Don't mix models trained with different `--n-components` values.  
Always use same setting:
- Training: `--n-components 6`
- Predicting: `--n-components 6`

### Problem: Model can't find config.yaml

**Solution**: Config is auto-saved next to model:
- Model: `./models/classifier.pkl`
- Config: `./models/config.yaml`

Both must be in same directory for `classify` command.

---

## Next Steps

1. **Run clustering** on your locations
2. **Review visualizations** - assess quality
3. **Label clusters** based on acoustic patterns
4. **Organize training data** in subdirectories
5. **Train supervised model** on labeled data
6. **Deploy** for new recordings

For detailed commands:

```bash
python -m src cluster --help
python -m src train --help
python -m src classify --help
```

---

## References

- **K-Means**: Scikit-learn KMeans documentation
- **Silhouette Score**: -1 (bad) to +1 (good)
- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **PCA**: Used for 2D visualization of 48-D feature space

---

## Questions?

If clustering looks wrong:
1. Check console output for warnings
2. Review silhouette plots
3. Try different k values
4. Verify audio quality and duration
5. Consider NMF component count (`--n-components`)

Good luck! 🎵🔊

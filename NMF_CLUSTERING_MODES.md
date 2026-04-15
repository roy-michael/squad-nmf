# NMF Clustering Modes - Advanced Pattern Discovery

**Latest Feature**: Multiple clustering strategies for discovering patterns at different levels of audio analysis.

---

## Overview

The clustering system now supports **4 distinct clustering modes**, each discovering different types of patterns in underwater recordings:

1. **Features Mode** (default) - Aggregate statistical patterns
2. **Basis Mode** - Frequency pattern discovery
3. **Activation Mode** - Temporal pattern discovery
4. **Combined Mode** - Complete NMF representation

---

## Clustering Modes Explained

### 1. **Features Mode** (Aggregated)

**Default behavior. Clusters entire audio files based on aggregate NMF statistics.**

```bash
python -m src cluster --input-dir ./data --cluster-mode features --n-clusters 3
```

**Feature Dimension**: 48 (6 NMF components × 8 statistical moments)

**What it clusters**:
- Mean, std, skewness, kurtosis of each NMF basis function (W)
- Mean, std, skewness, kurtosis of each NMF activation (H)
- Normalized to [0, 1]

**When to use**:
- Quick overview of file groupings
- Each file = one cluster assignment
- Good for initial exploration
- Fastest method (smallest feature space)

**Example output**:
```
Clustering Results:
  Silhouette Score: +0.0795
  Davies-Bouldin Index: 0.917
  
  Cluster 0: 2 files (similar overall spectral character)
  Cluster 1: 1 file (unique aggregate statistics)
  Cluster 2: 2 files (similar pattern)
```

---

### 2. **Basis Mode** (Frequency Patterns)

**Clusters the NMF basis functions (W) - frequency components.**

```bash
python -m src cluster --input-dir ./data --cluster-mode basis --n-clusters 3
```

**Feature Dimension**: 768 (128 mel bins × 6 components, flattened)

**What it clusters**:
- Raw W matrix from NMF (shape: 128 mel_bins × 6 components)
- Flattened to 768 features per file
- Captures spectral structure differences

**Physical meaning**:
- W matrix = learned frequency patterns
- Each row = response at one mel-frequency band
- Clustering W finds files with similar spectral "signatures"

**When to use**:
- Discover frequency-based similarities
- Find files with similar dominant frequencies
- Detect equipment/sensor variations
- Identify frequency-selective sources

**Example comparison**:
```
Features mode: Files grouped by overall envelope
Basis mode: Files grouped by which frequencies are emphasized
```

**Clinical interpretation**:
- Similar W matrices = similar frequency distributions
- Different W matrices = different frequency content
- Useful for identifying vehicle types by acoustic signature

---

### 3. **Activation Mode** (Temporal Patterns)

**Clusters the NMF activations (H) - temporal dynamics.**

```bash
python -m src cluster --input-dir ./data --cluster-mode activation --n-clusters 3
```

**Feature Dimension**: 6 (mean of H across time steps)

**What it clusters**:
- H matrix from NMF (shape: 6 components × time_steps)
- Takes mean across time: 6 components → 6 features
- Captures temporal dynamics averaged

**Physical meaning**:
- H matrix = activation strength over time
- Each row = how active one component is throughout recording
- Clustering H finds files with similar temporal dynamics

**When to use**:
- Discover temporal behavior patterns
- Find files with similar activation profiles
- Detect transient vs. continuous patterns
- Identify rhythm or pulse characteristics

**Silhouette comparison (typically higher than features mode)**:
```
Features mode: Silhouette +0.0795
Activation mode: Silhouette +0.1476  ← Better separation!
```

**Why activation performs better**:
- Fewer features (6 vs 48) → less noise
- Less curse of dimensionality
- Focuses on core temporal dynamics

---

### 4. **Combined Mode** (Complete NMF)

**Uses both W and H matrices together - complete representation.**

```bash
python -m src cluster --input-dir ./data --cluster-mode combined --n-clusters 3
```

**Feature Dimension**: 774 (768 from W + 6 from H)

**What it clusters**:
- Stacked: W flattened (768) + H mean (6)
- Combines spectral AND temporal information
- Most comprehensive representation

**When to use**:
- When you want to consider both frequency AND temporal patterns
- Most robust clustering (uses all information)
- When you have sufficient samples (avoid overfitting with many features)
- Hybrid analysis combining all NMF information

**Trade-offs**:
- ✅ Most complete information
- ✅ Often best separation quality
- ❌ Highest dimensionality (curse of dimensionality with small datasets)
- ❌ Slower computation
- ⚠️ May overfit with <10 files

---

## Worker Limiting

### Control Parallelization

```bash
# Use all CPU cores (default)
python -m src cluster --input-dir ./data --n-jobs -1

# Sequential processing (for debugging)
python -m src cluster --input-dir ./data --n-jobs 1

# Use 4 specific cores
python -m src cluster --input-dir ./data --n-jobs 4

# Use half available cores
python -m src cluster --input-dir ./data --n-jobs 8
```

### When to limit workers:

**Use `-1` (all cores)**:
- Large datasets (10+ files)
- Workstations/servers with many cores
- Time-sensitive analysis

**Use `1` (sequential)**:
- Debugging
- Resource-constrained systems
- Deterministic/reproducible results
- Single-threaded requirement

**Use `N` (specific count)**:
- Share system with other processes
- Memory-constrained containers
- Load balancing

---

## Performance Comparison

### Feature Space Sizes

| Mode | Features | Basis | Activation | Combined |
|------|----------|-------|------------|----------|
| **Dimension** | 48 | 768 | 6 | 774 |
| **Compute** | Fast | Medium | Fast | Slowest |
| **Memory** | Low | High | Low | Very high |
| **Separation** | Good | Variable | Often best | Good |

### Silhouette Scores (5 files test data)

```
Features mode:   +0.0795  (baseline)
Basis mode:      +0.0804  (similar)
Activation mode: +0.1476  ← BEST (2x improvement!)
Combined mode:   +0.0809  (similar to features)
```

**Key insight**: Activation mode often performs BEST with fewer features!

---

## Use Cases by Mode

### Frequency Pattern Discovery (Basis)
```bash
# Find files with similar engine signatures
python -m src cluster --input-dir ./recordings --cluster-mode basis --n-clusters 5

# Result: Groups by dominant frequencies
# Typical: AUV propellers, submarine engines, surface vessels
```

### Temporal Pattern Discovery (Activation)
```bash
# Find files with similar temporal behavior
python -m src cluster --input-dir ./recordings --cluster-mode activation --n-clusters 3

# Result: Groups by how components activate over time
# Typical: Continuous vs pulsed sounds, startup transients
```

### Complete Analysis (Combined)
```bash
# Comprehensive clustering using all NMF info
python -m src cluster --input-dir ./recordings --cluster-mode combined --n-clusters 4

# Result: Groups by both spectral AND temporal character
# Most robust but needs larger datasets
```

### Resource-Constrained (Sequential)
```bash
# Slower but lower memory footprint
python -m src cluster --input-dir ./recordings --n-jobs 1 --cluster-mode basis

# Useful on embedded systems or shared servers
```

---

## Practical Workflow

### Step 1: Quick Overview (Features)
```bash
python -m src cluster --input-dir ./recordings \
  --cluster-mode features \
  --n-clusters 3 \
  --plot-dir ./analysis_features
```

### Step 2: Frequency Analysis (Basis)
```bash
python -m src cluster --input-dir ./recordings \
  --cluster-mode basis \
  --n-clusters 3 \
  --plot-dir ./analysis_basis
```

### Step 3: Temporal Analysis (Activation)
```bash
python -m src cluster --input-dir ./recordings \
  --cluster-mode activation \
  --find-optimal \
  --plot-dir ./analysis_activation
```

### Step 4: Choose Best Mode
- Compare silhouette scores across modes
- Pick mode with clearest separation
- Verify with domain knowledge

### Step 5: Scale & Deploy
```bash
# Use best mode with full dataset
python -m src cluster --input-dir ./large_dataset \
  --cluster-mode activation \
  --n-clusters 5 \
  --n-jobs -1 \
  --plot-dir ./final_results
```

---

## Technical Details

### NMF Decomposition

Standard Non-negative Matrix Factorization:
```
Mel Spectrogram (128 × T) ≈ W (128 × 6) × H (6 × T)

where:
  W = basis functions (frequency patterns)
  H = activations (temporal coefficients)
  T = time steps
```

### Feature Extraction Per Mode

**Features**: Statistical moments of W and H
```python
features = [
    mean(W[:, 0]), std(W[:, 0]), skew(W[:, 0]), kurtosis(W[:, 0]),  # Component 0 basis
    mean(H[0, :]), std(H[0, :]), skew(H[0, :]), kurtosis(H[0, :]),  # Component 0 activation
    # ... repeat for 5 more components
]  # Total: 48 features
```

**Basis**: Flattened W matrix
```python
features = W.flatten()  # Shape: (128 * 6,) = 768 features
```

**Activation**: Mean H
```python
features = np.mean(H, axis=1)  # Shape: (6,) = 6 features
```

**Combined**: Concatenated
```python
features = np.concatenate([W.flatten(), np.mean(H, axis=1)])  # Shape: 774
```

---

## Troubleshooting

### "Silhouette Score: NaN"
- Occurs when all samples assign to one cluster
- Try `--find-optimal` to auto-detect best k
- Increase `--n-clusters` if k=2 fails

### Poor separation in Features mode?
- Try Activation mode (usually better!)
- Use `--find-optimal` to find better k
- Ensure recordings have sufficient variety

### Memory issues with Basis mode?
- Reduce `--n-jobs` to limit parallel workers
- Use `--n-jobs 1` for sequential (minimal memory)
- Try Activation mode instead (6 features vs 768)

### Slow processing?
- Use `--n-jobs -1` (use all cores)
- Try Activation mode (fastest)
- Pre-filter recordings to essential ones

---

## References

- **NMF**: https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf
- **Silhouette Score**: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- **K-Means**: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

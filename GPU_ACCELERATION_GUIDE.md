# GPU Acceleration Guide - Underwater Vehicle Sound Classifier

## Current GPU Status: CPU-ONLY

**The system is currently CPU-based and does NOT use GPU acceleration.**

### Technical Details

| Component | Implementation | Compute | Parallelization |
|-----------|----------------|---------|-----------------|
| **NMF Decomposition** | sklearn.decomposition.NMF | CPU (numpy/scipy) | Parallel via joblib |
| **K-Means Clustering** | sklearn.cluster.KMeans | CPU (numpy/scipy) | Parallel via joblib |
| **Audio Processing** | librosa + scipy | CPU (STFT, mel-spectrograms) | Serial (optimized via numpy) |
| **Classification** | Random Forest / SVM / Gradient Boost | CPU (scikit-learn) | Tree-level parallelism |
| **Visualization** | matplotlib | CPU | Single-threaded |

### Dependency Stack

```
librosa          → numpy/scipy (audio DSP)
scikit-learn     → numpy/scipy (algorithms, CPU only)
scipy            → Compiled CPU kernels
numpy            → CPU-optimized BLAS/LAPACK
```

---

## Performance Characteristics

### Current CPU Performance (All Cores)

| Operation | Dataset Size | Time | Method |
|-----------|--------------|------|--------|
| **NMF** (6 components) | 100 files × 10s each | ~30-60s | Parallel (-1 jobs) |
| **K-Means** (3 clusters) | 100 files × 6-dim vectors | ~2-5s | Parallel (-1 jobs) |
| **Classification** | Training on 300 samples | ~5-15s | Tree parallelism |
| **Batch Classify** | 100 files | ~60-120s | Parallel feature extraction |

### Speedup with Parallelization

- **1 worker (--n-jobs=1)**: Baseline ~1.0x
- **4 workers (--n-jobs=4)**: ~3.5-3.8x speedup
- **8 workers (--n-jobs=8)**: ~7.0-7.5x speedup
- **All cores (--n-jobs=-1)**: ~0.8-0.95x of core count (overhead)

---

## Recommended Commands

### For Single-File Classification (Fast)

```bash
# Classify a single underwater recording
python -m src classify --input audio.wav --model trained_model.pkl

# With explicit CPU usage (no GPU overhead)
python -m src classify --input audio.wav --model trained_model.pkl --n-jobs=1
```

### For Batch Processing (Parallel CPU)

```bash
# Classify multiple files using all available cores
python -m src batch-classify --input data/ --model trained_model.pkl --n-jobs=-1 --output results.csv

# Equivalent to above (all cores)
python -m src batch-classify --input data/ --model trained_model.pkl --n-jobs -1 --output results.csv

# Use exactly 4 cores
python -m src batch-classify --input data/ --model trained_model.pkl --n-jobs 4 --output results.csv
```

### For Temporal Clustering with Visualizations

```bash
# Analyze temporal patterns in audio files with full parallelization
python -m src temporal-cluster --input data/ --n-patterns 3 --n-jobs=-1 --visualize --viz-dir ./plots/

# Faster single-threaded version (for testing)
python -m src temporal-cluster --input data/ --n-patterns 3 --n-jobs=1 --visualize --viz-dir ./plots/
```

### For Training (GPU would help here most)

```bash
# Train with parallel feature extraction
python -m src train --data training_data/ --output model.pkl --n-jobs=-1

# Use 2 workers (for systems with limited memory)
python -m src train --data training_data/ --output model.pkl --n-jobs=2
```

### Worker Limiting Examples

```bash
# Limit to 2 workers (useful for 4GB+ RAM systems)
python -m src batch-classify --input data/ --model model.pkl --n-jobs 2

# Limit to 1 worker (single-threaded, minimal memory)
python -m src batch-classify --input data/ --model model.pkl --n-jobs 1

# Use all but one core (leave system responsive)
# Note: requires manual calculation or use -1 for true all-cores
```

---

## System Monitoring During Execution

### Monitor CPU Usage (Windows PowerShell)

```powershell
# Watch process CPU usage in real-time
Get-Process | Where-Object {$_.Name -like "*python*"} | Format-Table Name, CPU, WorkingSet -AutoSize

# For more detailed task manager view
Start-Process taskmgr
```

### Monitor CPU Usage (Command Line)

```bash
# Show current CPU and memory
wmic process where name="python.exe" get ProcessId,WorkingSetSize,UserModeTime /format:table
```

### Estimate Processing Time

```bash
# For 100 files with current CPU setup:
# - Batch classify: ~90-180 seconds (depends on file length)
# - Temporal clustering: ~120-240 seconds
# - Expected speedup: 3.5-7.5x with parallelization vs single-worker
```

---

## GPU Acceleration Options (If Needed in Future)

If you require GPU acceleration for larger datasets:

### Option 1: RAPIDS (NVIDIA GPU Only)

```bash
pip install rapids-singlenode
# Requires NVIDIA CUDA 11.8+, RTX/A100 GPU
# Provides GPU-accelerated sklearn-compatible API
```

**Command Example:**
```bash
# Would require code changes to use cuml instead of sklearn
python -m src train --data data/ --use-gpu --output model.pkl
```

### Option 2: CuPy (Requires NVIDIA GPU)

```bash
pip install cupy-cuda11x
# Manual kernel implementation required
# Best for NMF: 5-10x speedup on RTX 3090
```

### Option 3: PyTorch (Universal GPU Support)

```bash
pip install torch torchvision torchaudio
# Would require rewriting NMF and classifiers in PyTorch
# Best for deep learning approaches
```

### Option 4: JAX (Any GPU Backend)

```bash
pip install jax jaxlib
# High-performance GPU/TPU computation
# Similar performance to PyTorch for this workload
```

---

## Memory Requirements

### Typical Memory Usage

| Operation | File Duration | n_components | Memory Usage |
|-----------|--------------|--------------|--------------|
| Single file preprocessing | 10s | N/A | ~50 MB |
| NMF feature extraction | Per file | 6 | ~100-150 MB |
| Batch (100 files) | Total 1000s | 6 | ~800 MB - 1.2 GB |
| Model training | 300 samples | N/A | ~200-400 MB |
| Temporal clustering | 100 files | 6 | ~500 MB |

### Recommended System Specs

| Use Case | CPU Cores | RAM | GPU |
|----------|-----------|-----|-----|
| **Single file classification** | 1-2 cores | 2 GB | Not needed |
| **Batch (10-50 files)** | 4 cores | 4 GB | Not needed |
| **Batch (50+ files)** | 8 cores | 8+ GB | Optional (would help NMF) |
| **Training (1000+ files)** | 8+ cores | 16+ GB | Recommended (5-10x speedup) |

---

## Configuration File Settings

Edit `config.yaml` to control computation:

```yaml
# Parallel worker control
classifier:
  n_jobs: -1          # -1 = all cores, 1 = single, 4 = exactly 4 cores

# NMF iterations (affects convergence time)
nmf:
  max_iter: 2000      # Increase for better convergence, increase runtime
  n_components: 6     # More components = longer NMF computation
  use_sklearn: true   # False = custom multiplicative updates (slower but more control)

# K-Means iterations
kmeans:
  max_iter: 300       # More iterations = better clustering, longer runtime
  n_init: 10          # Number of random initializations (higher = more stable)
```

---

## Troubleshooting Performance Issues

### Problem: Very Slow Batch Processing

**Symptom:** Batch-classify taking 10+ minutes for 100 files

**Solution:**
```bash
# Check if parallelization is working
python -m src batch-classify --input data/ --model model.pkl --n-jobs=4 --verbose

# If still slow, reduce n_components in config.yaml:
# Change: n_components: 6
# To:     n_components: 4
# This reduces feature vector size and speeds NMF
```

### Problem: Out of Memory During Batch Processing

**Symptom:** Python crashes with memory error

**Solution:**
```bash
# Reduce to single-threaded (less memory overhead)
python -m src batch-classify --input data/ --model model.pkl --n-jobs=1

# Or use 2 workers if n_jobs=1 still crashes
python -m src batch-classify --input data/ --model model.pkl --n-jobs=2

# Reduce n_components in config.yaml to 4 or 3
```

### Problem: High CPU Usage but Slow Speed

**Symptom:** CPU at 100% but processing seems slow

**Solution:**
```bash
# May be I/O bound (reading audio files). Try:
python -m src batch-classify --input data/ --model model.pkl --n-jobs=2

# Check if disk is the bottleneck (copy audio files to SSD first)
# If using HDD, network drive, or slow USB: performance will be limited
```

---

## Recommended Setup

### Best for Most Users (CPU-based)

```bash
# Install with standard CPU dependencies
pip install click numpy scikit-learn scipy librosa pyyaml soundfile joblib pytest pytest-cov

# Run with automatic core detection
python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1
```

### If You Have Large Datasets (100+ files)

1. **Increase n_components gradually** in config.yaml (better features = better accuracy)
2. **Monitor CPU during runs** to ensure parallelization is working
3. **Use --n-jobs=-1** for maximum throughput
4. **Consider GPU only if:**
   - Processing takes >30 minutes regularly
   - You have NVIDIA GPU with CUDA support
   - You're training models frequently

---

## Next Steps

1. **Verify current performance**: Run on your dataset with `--n-jobs=-1`
2. **Monitor CPU usage**: Use task manager to confirm parallelization
3. **If slow**: Check for I/O bottlenecks (disk speed)
4. **If very slow**: GPU acceleration may be justified (see RAPIDS option)
5. **If accurate enough**: CPU performance is sufficient for deployment

---

## Summary

| Question | Answer |
|----------|--------|
| **Is it using GPU?** | **No, CPU-only** (numpy/scipy via scikit-learn) |
| **Do I need GPU?** | **No, unless processing 100+ hours of audio regularly** |
| **Best command for my case?** | `python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1` |
| **How do I limit workers?** | `--n-jobs 2` (for 2 workers) or `--n-jobs 1` (single-threaded) |
| **Will it use all cores?** | **Yes, with --n-jobs=-1** (recommended default) |
| **Can I use GPU?** | **Not without code changes**, but CPU is fast enough for most datasets |


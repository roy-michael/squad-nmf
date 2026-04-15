# Command Reference - Quick Guide

## 🚀 Most Common Commands

### **Single File Classification**
```bash
python -m src classify --input audio.wav --model trained_model.pkl
```

### **Batch Classify (Parallel - Recommended)**
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1 --output results.csv
```

### **Batch Classify with Worker Limit**
```bash
# Use exactly 2 workers (useful for limited memory)
python -m src batch-classify --input data/ --model model.pkl --n-jobs 2 --output results.csv

# Use exactly 4 workers
python -m src batch-classify --input data/ --model model.pkl --n-jobs 4 --output results.csv

# Use 1 worker (single-threaded, lowest memory)
python -m src batch-classify --input data/ --model model.pkl --n-jobs 1 --output results.csv
```

### **Train New Model**
```bash
# Data directory should have: training_data/ClassName1/*.wav, training_data/ClassName2/*.wav
python -m src train --data training_data/ --output model.pkl --n-jobs=-1
```

### **Temporal Clustering (Find Multiple Patterns)**
```bash
python -m src temporal-cluster --input data/ --n-patterns 3 --n-jobs=-1 --visualize --viz-dir ./plots/
```

### **Unsupervised K-Means Clustering**
```bash
python -m src cluster --input data/ --n-clusters 3 --output clusters.json
```

---

## 📊 Command Structure

### Full Command Format with All Options
```bash
python -m src COMMAND [OPTIONS]
```

---

## 🎯 Worker Limiting Guide

| Use Case | Command | Speed | Memory |
|----------|---------|-------|--------|
| **Single file, testing** | `--n-jobs 1` | Slow | ~50-100 MB |
| **Batch, 4GB RAM system** | `--n-jobs 2` | Medium | ~1-2 GB |
| **Batch, 8GB RAM system** | `--n-jobs 4` | Fast | ~3-4 GB |
| **Batch, 16GB+ RAM system** | `--n-jobs=-1` | Fastest | ~6-8 GB |

### Worker Count Recommendations

**`--n-jobs 1`** (Single Worker)
- Best for: Low-memory systems (<4GB), testing, debugging
- Speed: Slowest but most stable
- Memory: ~50-100 MB overhead

**`--n-jobs 2`** (Two Workers)
- Best for: Systems with 4-8GB RAM
- Speed: ~1.8-2x faster than single worker
- Memory: ~500 MB-1.5 GB total

**`--n-jobs 4`** (Four Workers)
- Best for: Systems with 8GB+ RAM, typical workstations
- Speed: ~3.5-3.8x faster than single worker
- Memory: ~2-3 GB total

**`--n-jobs=-1`** (All Cores)
- Best for: High-end systems with 16GB+ RAM and 8+ cores
- Speed: ~7-8x faster than single worker (7.5-8 cores)
- Memory: ~6-8 GB total

---

## 🔧 Output Options

### Save Results as CSV
```bash
python -m src batch-classify --input data/ --model model.pkl --output results.csv --n-jobs=-1
```

### Save Visualizations
```bash
python -m src temporal-cluster --input data/ --n-patterns 3 --visualize --viz-dir ./plots/
```

### Save Cluster JSON
```bash
python -m src cluster --input data/ --n-clusters 3 --output clusters.json
```

---

## 🔍 Verbose/Debug Mode

Add `--verbose` to any command for detailed logging:

```bash
python -m src classify --input audio.wav --model model.pkl --verbose

python -m src batch-classify --input data/ --model model.pkl --n-jobs=2 --verbose

python -m src train --data training_data/ --output model.pkl --n-jobs=-1 --verbose
```

---

## 📈 Performance Tuning

### For Faster Processing (Recommended)
```bash
# Use all cores with increased NMF iterations in config.yaml
python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1
```

### For Lower Memory Usage
```bash
# Use single worker
python -m src batch-classify --input data/ --model model.pkl --n-jobs 1
```

### For Better Accuracy (Slower)
Edit `config.yaml`:
```yaml
nmf:
  n_components: 8        # Increased from 6 (more features)
  max_iter: 3000         # Increased from 2000
```

Then run:
```bash
python -m src train --data training_data/ --output model_improved.pkl --n-jobs=-1
```

---

## 📋 Example Workflows

### Workflow 1: Complete Training and Classification

```bash
# Step 1: Train model (parallelized)
python -m src train --data training_data/ --output my_model.pkl --n-jobs=-1

# Step 2: Classify new recordings (parallelized)
python -m src batch-classify --input new_recordings/ --model my_model.pkl --n-jobs=-1 --output results.csv

# Step 3: Analyze temporal patterns
python -m src temporal-cluster --input new_recordings/ --n-patterns 3 --n-jobs=-1 --visualize --viz-dir ./analysis/
```

### Workflow 2: Quick Testing (Low Resource)

```bash
# Step 1: Train on subset (single worker)
python -m src train --data training_data/ --output test_model.pkl --n-jobs 1

# Step 2: Test on few files (single worker)
python -m src batch-classify --input test_recordings/ --model test_model.pkl --n-jobs 1 --output test_results.csv
```

### Workflow 3: Production Batch Processing

```bash
# Classify large dataset with optimal parallelization
python -m src batch-classify \
  --input /path/to/large/dataset/ \
  --model production_model.pkl \
  --n-jobs=-1 \
  --output production_results.csv \
  --verbose

# Then analyze results
python -m src temporal-cluster \
  --input /path/to/large/dataset/ \
  --n-patterns 5 \
  --n-jobs=-1 \
  --visualize \
  --viz-dir ./production_analysis/
```

---

## 🚨 Troubleshooting Commands

### If Running Out of Memory
```bash
# Try with 2 workers instead of all cores
python -m src batch-classify --input data/ --model model.pkl --n-jobs 2 --output results.csv
```

### If Processing Too Slow
```bash
# Check verbose output to see what's slow
python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1 --verbose --output results.csv

# If NMF is slow, reduce components in config.yaml
# If disk is slow, consider copying files to SSD first
```

### If Need to Monitor CPU Usage
Open PowerShell and run:
```powershell
Get-Process | Where-Object {$_.Name -like "*python*"} | Format-Table Name, CPU, WorkingSet -AutoSize
```

---

## 📚 GPU Information

**Current Status:** CPU-only (no GPU support)

**Why?** System uses scikit-learn (CPU) via numpy/scipy

**Do you need GPU?** Only if processing takes >30 minutes regularly

**Best command to start:** 
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1
```

For details, see **GPU_ACCELERATION_GUIDE.md**

---

## 📖 Full Help

See all available commands:
```bash
python -m src --help
```

Get help on specific command:
```bash
python -m src classify --help
python -m src batch-classify --help
python -m src train --help
python -m src temporal-cluster --help
```

---

## 💡 Quick Decision Matrix

| Question | Answer | Command |
|----------|--------|---------|
| Classify 1 file? | `classify` | `python -m src classify --input file.wav --model model.pkl` |
| Classify many files fast? | `batch-classify` with `-1 jobs` | `python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1` |
| Low memory (4GB)? | `batch-classify` with `2 jobs` | `python -m src batch-classify --input data/ --model model.pkl --n-jobs 2` |
| Train new model? | `train` with `-1 jobs` | `python -m src train --data train_data/ --output model.pkl --n-jobs=-1` |
| Find multiple patterns? | `temporal-cluster` | `python -m src temporal-cluster --input data/ --n-patterns 3 --n-jobs=-1 --visualize` |
| Debug/test mode? | Any command with `--verbose` | `python -m src classify --input file.wav --model model.pkl --verbose` |


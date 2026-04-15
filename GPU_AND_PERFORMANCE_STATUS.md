# GPU & Performance Status Summary

## Direct Answers to Your Questions

### ❓ "Is it using GPU?"
**Answer: NO** — The system is **CPU-only** and does NOT use GPU acceleration.

**Why?**
- All core algorithms use scikit-learn, which only runs on CPU
- numpy/scipy (underlying computation) are CPU-based
- No GPU drivers (CUDA/ROCm) are configured
- This is intentional for maximum compatibility

---

### ❓ "What should be the command to use?"

**Short Answer:**
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1
```

---

## 📋 Full Command Guide

### **For Classifying Multiple Files (RECOMMENDED)**

```bash
# Use ALL available cores (fastest)
python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1 --output results.csv
```

### **Worker Limiting Examples**

#### Use Exactly 2 Workers
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs 2 --output results.csv
```
**When:** System has 4-8GB RAM, you want to limit overhead

#### Use Exactly 4 Workers
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs 4 --output results.csv
```
**When:** System has 8GB+ RAM, typical laptop/desktop

#### Use 1 Worker (Single-Threaded)
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs 1 --output results.csv
```
**When:** System is low on memory (<4GB), debugging needed

---

## 🔧 The `--n-jobs` Parameter

| Value | Effect | Speed vs Single | Memory | When to Use |
|-------|--------|-----------------|--------|------------|
| `1` | Single-threaded | 1.0x (baseline) | Lowest | Low-RAM systems |
| `2` | 2 parallel workers | ~1.8x | Low-Medium | 4GB RAM systems |
| `4` | 4 parallel workers | ~3.5x | Medium | 8GB RAM systems |
| `-1` | All CPU cores | ~7.5x (8 cores) | High | 16GB+ systems |

### How to Set It

```bash
# Option 1: Command line (recommended, easy to change)
python -m src batch-classify --input data/ --model model.pkl --n-jobs 2

# Option 2: Edit config.yaml (permanent for all commands)
# Edit line in config.yaml:
#   classifier:
#     n_jobs: 2
```

---

## ⚡ Performance Expectations

### Processing Time Estimates

| Scenario | Files | Duration | Single Worker | 4 Workers | 8 Workers | Expected Output |
|----------|-------|----------|---------------|-----------|-----------|-----------------|
| Quick test | 10 files | 10s each | ~40s | ~15s | ~8s | results.csv |
| Medium batch | 50 files | 10s each | ~200s | ~60s | ~30s | results.csv |
| Large batch | 100 files | 10s each | ~400s | ~120s | ~60s | results.csv |
| Training | 300 samples | Various | ~60s | ~20s | ~12s | model.pkl |

---

## 🚀 Quick Start

### Option A: Maximum Speed
```bash
python -m src batch-classify --input recordings/ --model model.pkl --n-jobs=-1 --output results.csv
```

### Option B: Balanced (Recommended for Most Systems)
```bash
python -m src batch-classify --input recordings/ --model model.pkl --n-jobs=4 --output results.csv
```

### Option C: Low Memory Usage
```bash
python -m src batch-classify --input recordings/ --model model.pkl --n-jobs=1 --output results.csv
```

---

## 🎯 What Happens Behind the Scenes

When you run with `--n-jobs=-1`:

```
┌─────────────────────────────────────────────────────────┐
│ Main Process (Orchestrator)                             │
│ - Loads audio files                                     │
│ - Distributes to worker processes                       │
└─────────────────────────────────────────────────────────┘
        ↓         ↓         ↓         ↓
    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │Core1 │ │Core2 │ │Core3 │ │Core4 │
    │Worker│ │Worker│ │Worker│ │Worker│
    │      │ │      │ │      │ │      │
    │NMF   │ │NMF   │ │NMF   │ │NMF   │
    │Class │ │Class │ │Class │ │Class │
    │      │ │      │ │      │ │      │
    └──────┘ └──────┘ └──────┘ └──────┘
        ↓         ↓         ↓         ↓
┌─────────────────────────────────────────────────────────┐
│ Collect Results → Write CSV                             │
└─────────────────────────────────────────────────────────┘
```

**All cores process files in parallel** = ~4x speedup with 4 cores

---

## 📊 Monitoring Your Run

### Watch CPU Usage in Task Manager
```
1. Press Ctrl+Shift+Esc
2. Click "Performance" tab
3. Watch CPU% - should be near 100%
4. Watch Memory - should stay below system RAM
```

### Command Line Monitoring (PowerShell)
```powershell
# Real-time CPU and memory
Get-Process | Where-Object {$_.Name -like "*python*"} | Format-Table Name, CPU, WorkingSet -AutoSize
```

---

## ❗ Common Issues & Solutions

### Issue: Out of Memory Error

**Error Message:** `MemoryError` or system becomes unresponsive

**Solution:**
```bash
# Reduce workers
python -m src batch-classify --input data/ --model model.pkl --n-jobs 2 --output results.csv
```

### Issue: Very Slow Processing

**Symptom:** Even with all cores, processing is slow

**Diagnosis:**
```bash
# Add --verbose to see what's happening
python -m src batch-classify --input data/ --model model.pkl --n-jobs=4 --output results.csv --verbose
```

**Possible Causes:**
- ❌ Files are on slow USB/network drive → Copy to SSD first
- ❌ Disk I/O is bottleneck → Not helped by more workers
- ❌ NMF parameters too large → Reduce `n_components` in config.yaml

### Issue: Python Process Uses All Memory Then Crashes

**Solution:** Use fewer workers
```bash
python -m src batch-classify --input data/ --model model.pkl --n-jobs 1 --output results.csv
```

---

## 📚 Full Documentation

**For detailed information, see:**
- `GPU_ACCELERATION_GUIDE.md` — Complete performance guide
- `COMMAND_REFERENCE.md` — All available commands
- `TEMPORAL_CLUSTERING_GUIDE.md` — Finding multiple patterns
- `VISUALIZATION_INTERPRETATION_GUIDE.md` — Reading visualizations

---

## 🎓 Summary Table

| Question | Answer |
|----------|--------|
| **GPU Support?** | **NO** (CPU-only via sklearn) |
| **Multi-threaded?** | **YES** (parallel workers via joblib) |
| **Worker Limiting?** | **YES** (`--n-jobs 1`, `2`, `4`, or `-1`) |
| **Recommended Command?** | `python -m src batch-classify --input data/ --model model.pkl --n-jobs=-1` |
| **System Requirement?** | 4GB RAM minimum; 8GB+ recommended with parallelization |
| **Performance Speedup?** | ~3.5x-7.5x faster with parallelization vs single worker |

---

## 🚀 Next Steps

1. **Run with default CPU parallelization:**
   ```bash
   python -m src batch-classify --input recordings/ --model model.pkl --n-jobs=-1
   ```

2. **If fast enough:** Continue with this command

3. **If memory issues:** Reduce workers:
   ```bash
   python -m src batch-classify --input recordings/ --model model.pkl --n-jobs 2
   ```

4. **If you later need GPU:** Contact support — would require code changes to use RAPIDS or PyTorch

---

**File:** This summary explains the GPU status and provides recommended commands.
**Location:** D:\dev\squad-nmf\
**Updated:** Latest session
**Status:** Ready for production use (CPU-based)


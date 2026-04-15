# Understanding `--n-patterns` Parameter

## Quick Answer

**`--n-patterns`** specifies how many distinct **temporal patterns** (sound signatures) the system should discover within each audio file.

```bash
python -m src temporal-cluster --input data/ --n-patterns 3
                                               ↑
                                    Find 3 different time periods
```

---

## What Are "Patterns"?

A **pattern** is a distinct time period in your audio where a particular sound signature is active.

### Example: 30-second underwater recording

```
Timeline of Recording:
├─ 0-10s:   Background ocean noise + ship engine
│           Pattern A: Low-frequency propulsion
│
├─ 10-20s:  Ship accelerating + different frequency
│           Pattern B: Higher-frequency acceleration
│
└─ 20-30s:  New vehicle sound OR echo decay
            Pattern C: Different acoustic signature
```

**With `--n-patterns 3`**, the system would identify these 3 distinct time windows.

---

## How It Works

### Step 1: Extract NMF Features
For each file, compute NMF (6 basis functions) giving an **activation matrix**:
- Rows = Time frames (one per ~11ms)
- Columns = 6 NMF basis activations

### Step 2: Cluster Time Frames
K-means clustering groups similar time frames:
- **`--n-patterns 2`**: Find 2 temporal clusters (e.g., "silent periods" + "active sound")
- **`--n-patterns 3`**: Find 3 temporal clusters (e.g., "ambient" + "vehicle A" + "vehicle B")
- **`--n-patterns 4`**: Find 4 temporal clusters (more fine-grained breakdown)

### Step 3: Map to Time
Convert frame indices back to seconds:
```
Frame 1000 → ~11 seconds
Frame 2000 → ~22 seconds
```

---

## Choosing `--n-patterns` Value

### Recommended Values

| n-patterns | Best For | Example Use Case |
|-----------|----------|------------------|
| **1** | Single homogeneous sound | File with one vehicle throughout |
| **2** | Simple temporal split | Background + distinct event |
| **3** | Most common (DEFAULT) | Multiple vehicles or phase transitions |
| **4-5** | Fine-grained analysis | Complex multi-vehicle scenario |
| **6+** | Detailed temporal breakdown | Very complex soundscapes |

### Decision Tree

```
How complex is your soundscape?

Simple (one sound throughout)        → --n-patterns 1
Simple split (background + event)    → --n-patterns 2
Typical (2-3 distinct periods)       → --n-patterns 3 (DEFAULT)
Complex (many transitions)           → --n-patterns 4
Very complex (many overlaps)         → --n-patterns 5-6
```

---

## Real-World Examples

### Example 1: Submarine Passage (n-patterns 3)

```bash
python -m src temporal-cluster --input recordings/ --n-patterns 3 --visualize --viz-dir plots/
```

**Expected Result:**
```
File: submarine_passage.wav

Pattern_0 (Background/Ambient):
  - Time: 0.0s - 5.2s
  - Strength: 0.001 (very weak)
  - Components: [4, 5] (ambient frequencies)

Pattern_1 (Submarine Approaching):
  - Time: 5.2s - 12.8s
  - Strength: 0.045 (moderate)
  - Components: [0, 1] (low-frequency diesel)

Pattern_2 (Submarine Passing/Departing):
  - Time: 12.8s - 18.3s
  - Strength: 0.032 (moderate)
  - Components: [2, 3] (cavitation frequencies)
```

### Example 2: Simple Ambient Recording (n-patterns 1)

```bash
python -m src temporal-cluster --input recordings/ --n-patterns 1 --visualize --viz-dir plots/
```

**Expected Result:**
```
File: ambient.wav

Pattern_0 (Constant Ambient):
  - Time: 0.0s - 45.0s
  - Strength: 0.002
  - Components: [4, 5] (ambient throughout)
```

### Example 3: Complex Multi-Vehicle Scene (n-patterns 5)

```bash
python -m src temporal-cluster --input recordings/ --n-patterns 5 --visualize --viz-dir plots/
```

**Expected Result:**
```
File: busy_harbor.wav

Pattern_0: Ambient (0-5s)
Pattern_1: Ship A (5-12s)
Pattern_2: Ship B (12-18s, overlapping)
Pattern_3: Acceleration (18-22s)
Pattern_4: Transition/Fade (22-30s)
```

---

## Output Interpretation

When you run:
```bash
python -m src temporal-cluster --input data/ --n-patterns 3 --visualize --viz-dir plots/
```

### Output Files Generated

1. **temporal_analysis.json** - Results data
2. **plots/file1_timeline.png** - Temporal pattern timeline
3. **plots/file1_heatmap.png** - Activation heatmap
4. **plots/file1_summary.png** - Pattern summary statistics

### What Each Pattern Shows

For each pattern, you get:
- **Time Range** - When this pattern occurs (in seconds)
- **Active Components** - Which NMF bases are dominant
- **Strength** - Intensity (0=silent, 1=maximum)

---

## Guidelines

### Matching n-patterns to Your Data

**Rule of Thumb:** `n-patterns` = Expected number of distinct sound periods + 1

```
Scenario: Recording with 2 distinct vehicle sounds
Expected patterns: 2 vehicles + 1 background noise
Recommended: --n-patterns 3
```

### If Unsure: Start with Default

```bash
# Default is --n-patterns 3
python -m src temporal-cluster --input data/ --visualize --viz-dir plots/
```

This works well for:
- Multi-vehicle scenarios
- Sequential sound events
- Background + foreground separation

### Adjust Based on Results

If results look wrong:

**Too many small fragments?**
- Reduce `--n-patterns` (try 2 instead of 3)

**Merging distinct sounds?**
- Increase `--n-patterns` (try 4 instead of 3)

```bash
# Experiment: Try different values
python -m src temporal-cluster --input data/ --n-patterns 2 --visualize --viz-dir plots_2/
python -m src temporal-cluster --input data/ --n-patterns 4 --visualize --viz-dir plots_4/
# Compare results in plots_2/ and plots_4/
```

---

## Technical Details

### How Temporal Clustering Works

1. **Input**: Audio file → NMF extracts H matrix (6 × 15001 time frames)

2. **Processing**: K-means clusters the H matrix rows (time frames)
   - Transpose: (6 × 15001) → (15001 × 6)
   - K-means: Group 15001 time frames into N clusters (N = n-patterns)

3. **Output**: Temporal patterns with:
   - Time boundaries (when pattern starts/ends)
   - Dominant components (which NMF bases active)
   - Strength score (mean activation)

### Mathematical Interpretation

```
H matrix row at frame t = [h₁(t), h₂(t), h₃(t), h₄(t), h₅(t), h₆(t)]
                          ^activation levels of 6 NMF components^

K-means clusters similar rows:
- Rows with similar activation patterns → same cluster
- Each cluster = one temporal pattern
```

---

## Performance Impact

| n-patterns | Time (100 files) | Memory | Notes |
|-----------|------------------|--------|-------|
| 1 | ~60s | ~200MB | Fastest |
| 2 | ~70s | ~250MB | Minimal overhead |
| 3 | ~80s | ~300MB | Default, balanced |
| 5 | ~100s | ~400MB | More detailed |
| 8+ | ~130s+ | ~500MB+ | Slow, diminishing returns |

**Recommendation:** Stick with 2-3 for production use.

---

## Common Use Cases

### Use Case 1: Find Two Distinct Periods
```bash
python -m src temporal-cluster --input data/ --n-patterns 2
```
**Example:** "Before detection" vs "After detection"

### Use Case 2: Find Multiple Vehicles (Most Common)
```bash
python -m src temporal-cluster --input data/ --n-patterns 3
```
**Example:** "Ambient" vs "Vehicle A" vs "Vehicle B"

### Use Case 3: Detailed Multi-Event Analysis
```bash
python -m src temporal-cluster --input data/ --n-patterns 5
```
**Example:** "Background" + "3 vehicle types" + "transition"

### Use Case 4: Simple Presence/Absence
```bash
python -m src temporal-cluster --input data/ --n-patterns 2
```
**Example:** "Silent period" vs "Sound event"

---

## FAQ

### Q: How is n-patterns different from n-clusters?
**A:** 
- **`--n-patterns`** (temporal clustering): How many distinct time periods in ONE file
- **`--n-clusters`** (K-means clustering): How many groups of entire FILES

### Q: What if n-patterns is too high?
**A:** Each pattern becomes smaller/noisier. May find artificial separation in continuous sound.

### Q: What if n-patterns is too low?
**A:** Different sounds get merged into same pattern. May miss distinct vehicles.

### Q: Can I use n-patterns = 1?
**A:** Yes, if your file has single consistent sound throughout. But loses temporal information.

### Q: Recommended value?
**A:** Start with **3** (default), adjust based on visualization results.

---

## Quick Reference

```bash
# Simplest: Use default (n-patterns=3)
python -m src temporal-cluster --input data/ --visualize --viz-dir plots/

# Explicit n-patterns value
python -m src temporal-cluster --input data/ --n-patterns 2 --visualize --viz-dir plots/

# Experiment with multiple values
python -m src temporal-cluster --input data/ --n-patterns 2 --visualize --viz-dir plots_2/
python -m src temporal-cluster --input data/ --n-patterns 4 --visualize --viz-dir plots_4/
```

---

## Summary

| Aspect | Explanation |
|--------|-------------|
| **What is it?** | Number of temporal clusters (distinct sound periods) to find per file |
| **Default value** | 3 |
| **Range** | 1-8+ (diminishing returns beyond 5) |
| **Recommendation** | 2-3 for most underwater recordings |
| **Adjust if** | Results show too many fragments (reduce) or merged sounds (increase) |

**Default command:**
```bash
python -m src temporal-cluster --input data/ --n-patterns 3 --visualize --viz-dir plots/
```


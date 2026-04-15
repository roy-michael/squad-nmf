# Temporal Clustering Guide - Discovering Multiple Sound Patterns Within Audio Files

**New Feature**: Temporal clustering enables discovery of multiple distinct sound patterns active within individual audio recordings.

---

## Why Temporal Clustering?

### The Problem
Traditional clustering assigns each audio **file** to a single cluster. But real underwater recordings often contain:

- **Multiple vehicles** passing through sequentially
- **Mixed ambient noise** and vehicle sounds simultaneously  
- **Different propulsion modes** (acceleration, cruising, stopped)
- **Echolocation events** mixed with propulsion noise

**Example**: A 30-second recording might contain:
- **0-10s**: Ambient ocean noise
- **10-20s**: Ship propulsion at constant speed
- **20-30s**: Different vehicle sound or echo

Traditional clustering would assign the entire file to one cluster, losing this temporal structure.

### The Solution
**Temporal clustering** discovers these patterns by analyzing how the **NMF activations change over time**.

---

## How It Works

### Step 1: NMF Decomposition
First, the system computes NMF on the full audio file:
- Input: Mel spectrogram (128 frequencies × 15001 time frames)
- Output: 
  - **W matrix** (128 × 6): Spectral basis functions (frequency patterns)
  - **H matrix** (6 × 15001): Activations over time (which bases are active at each moment)

### Step 2: Temporal Pattern Clustering
Instead of clustering entire files, we cluster **time frames** from the H matrix:

1. **Transpose H**: Convert (6 components × 15001 frames) → (15001 frames × 6 components)
   - Each row = one moment in time
   - Each column = activation level of one NMF component

2. **Cluster time frames**: K-means on transposed H
   - Groups similar time frames together
   - Discovers when different "sound signatures" occur
   - Default: K=2 clusters (background pattern + distinct event)

3. **Map back to time**: Convert frame indices to seconds
   - Frame index → Time using: `time_seconds = frame_index × hop_length / sample_rate`
   - Shows exact time range for each pattern

### Step 3: Pattern Characterization
For each temporal cluster, the system reports:
- **Time range** (start and end time in seconds)
- **Active components** (which NMF bases dominate this pattern)
- **Strength** (mean activation level: 0=inactive, 1=maximum)

---

## Usage

### Basic Command
```bash
python -m src temporal-cluster \
  --input-dir ./recordings \
  --n-patterns 2 \
  --n-jobs -1
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | Required | Directory containing WAV files |
| `--n-patterns` | 3 | Number of temporal clusters per file |
| `--output-json` | `temporal_analysis.json` | Output results file |
| `--n-jobs` | -1 | Parallel workers (-1 = all cores, 1 = sequential) |
| `--verbose` | False | Enable debug logging |

### Example Output

**JSON Results** (`temporal_analysis.json`):
```json
{
  "RBW6737_20250723_094900.wav": {
    "file": "RBW6737_20250723_094900.wav",
    "time_frames_analyzed": 15001,
    "temporal_clusters": 2,
    "patterns": {
      "Pattern_0": {
        "time_range": "0.00s - 174.15s",
        "active_components": [4, 5],
        "strength": 0.0005
      },
      "Pattern_1": {
        "time_range": "145.04s - 165.09s",
        "active_components": [0, 2],
        "strength": 0.0693
      }
    }
  },
  "RBW6737_20250723_095000.wav": {
    "file": "RBW6737_20250723_095000.wav",
    "time_frames_analyzed": 15001,
    "temporal_clusters": 2,
    "patterns": {
      "Pattern_0": {
        "time_range": "0.00s - 174.15s",
        "active_components": [1, 4],
        "strength": 0.0011
      },
      "Pattern_1": {
        "time_range": "22.55s - 22.93s",
        "active_components": [0, 5],
        "strength": 0.2121
      }
    }
  }
}
```

### Console Output
```
[*] Analyzing ./data/Location_2307 for 2 temporal patterns per file...

Temporal Clustering Results:

File: RBW6737_20250723_094900.wav
  Time frames analyzed: 15001
  Patterns found: 2
    Pattern_0:
      Time: 0.00s - 174.15s
      Active components: [4, 5]
      Strength: 0.0005
    Pattern_1:
      Time: 145.04s - 165.09s
      Active components: [0, 2]
      Strength: 0.0693

[OK] Saved temporal analysis: temporal_analysis.json

Interpretation Guide:
- Each pattern represents a distinct sound signature within the file
- Time range shows when each pattern was active
- Active components indicate which NMF bases contribute to each pattern
- Strength indicates how dominant this pattern is (0-1 scale)

Example:
  File contains Pattern_0 (0.0s-15.0s) and Pattern_1 (15.0s-30.0s)
  => Likely two different vehicles or noise sources
```

---

## Interpreting Results

### Pattern Strength
- **Strength 0.0-0.05**: Background/ambient noise pattern
- **Strength 0.05-0.15**: Moderate vehicle sound
- **Strength 0.15+**: Strong, distinct vehicle signature

### Active Components
The NMF components that dominate each pattern:
- If same components across patterns → **different time windows of same vehicle**
- If different components → **different vehicle types or noise sources**

### Time Ranges
- **Pattern_0 covering entire file (0s - end)**: Weak background signature
- **Pattern_1 small time window**: Distinct event (vehicle passing, echo, noise burst)
- **Multiple large time windows**: Multiple vehicles active sequentially

---

## Real-World Examples

### Example 1: Single Vehicle Passing
```
Recording: 30 seconds total
Pattern_0: 0.0s - 30.0s, strength 0.001 → Background ambient noise
Pattern_1: 8.0s - 12.5s, strength 0.15 → Vehicle passing overhead
Pattern_2: 15.0s - 20.0s, strength 0.08 → Echo or secondary reflection
```
**Interpretation**: One vehicle detected with clear event window

### Example 2: Multiple Vehicles Overlapping
```
Recording: 60 seconds total
Pattern_0: 0.0s - 60.0s, strength 0.002 → Persistent background
Pattern_1: 5.0s - 20.0s, strength 0.12 → Vehicle A (propulsion noise)
Pattern_2: 18.0s - 35.0s, strength 0.18 → Vehicle B (overlaps with A)
Pattern_3: 40.0s - 60.0s, strength 0.09 → Vehicle C (later passage)
```
**Interpretation**: At least 3 distinct vehicles with overlapping periods

### Example 3: Mixed Noise Environment
```
Recording: 120 seconds total
Pattern_0: 0.0s - 120.0s, strength 0.008 → Ocean ambient (wind, currents)
Pattern_1: 25.0s - 35.0s, strength 0.20 → Strong vehicle signature
Pattern_2: 45.0s - 55.0s, strength 0.05 → Weak distant sound
Pattern_3: 70.0s - 120.0s, strength 0.03 → Second background component
```
**Interpretation**: One clear vehicle; background has two distinct noise characteristics

---

## Choosing Number of Patterns

### Rules of Thumb
- **n_patterns=2** (default): Separates main signal from background
  - Pattern_0: Background ambient noise
  - Pattern_1: Distinct vehicle sound
  
- **n_patterns=3**: Finds main vehicle + variant modes
  - Pattern_0: Background
  - Pattern_1: Vehicle cruising
  - Pattern_2: Vehicle acceleration or echo
  
- **n_patterns=4+**: For complex multi-vehicle scenarios
  - Pattern_0: Background
  - Pattern_1-3: Different vehicles or propulsion modes
  - Pattern_4+: Fine details (harmonics, echoes, etc.)

### Diminishing Returns
More patterns don't always help:
- With only 2-3 distinct sounds, requesting 5+ patterns causes K-means to artificially subdivide
- Look for clusters with very similar time ranges → sign of over-clustering
- **Best practice**: Start with n_patterns=2, increase only if needed

---

## Performance & Parallel Processing

### Speed Characteristics
- **NMF extraction**: ~6-8 seconds per file (CPU-bound)
- **Temporal clustering**: <0.1 seconds per file (fast K-means on time frames)
- **Bottleneck**: NMF computation, not clustering

### Using --n-jobs

```bash
# Parallel processing (default, uses all cores)
python -m src temporal-cluster --input-dir ./data --n-jobs -1

# Sequential (useful for debugging)
python -m src temporal-cluster --input-dir ./data --n-jobs 1

# Limited workers (e.g., 4 cores max)
python -m src temporal-cluster --input-dir ./data --n-jobs 4
```

### Expected Performance
On a 28-core system with 5 files:
- Sequential (n_jobs=1): ~35-40 seconds
- Parallel (n_jobs=-1): ~6-8 seconds (**5x speedup**)
- Each core processes independently → near-linear scaling

---

## Advanced: Custom Analysis

### Scripting with Temporal Results
```python
import json

with open('temporal_analysis.json') as f:
    results = json.load(f)

# Find files with strong vehicle signatures (strength > 0.1)
for filename, data in results.items():
    for pattern_name, pattern in data['patterns'].items():
        if pattern['strength'] > 0.1:
            print(f"{filename}: {pattern_name} is strong")
            print(f"  Time: {pattern['time_range']}")
            print(f"  Components: {pattern['active_components']}")
```

### Combining with Other Modes
```bash
# First: Unsupervised K-means on full files (find file groups)
python -m src cluster --input-dir ./data --n-clusters 4

# Then: Temporal analysis (find patterns within each group)
python -m src temporal-cluster --input-dir ./data --n-patterns 3

# Result: Multi-level understanding of data structure
```

---

## Troubleshooting

### Pattern_0 covers entire file with low strength
**Normal**. This is background/ambient noise that persists throughout. Look for Pattern_1+.

### All patterns have similar time ranges
**Over-clustering**. Reduce `--n-patterns`. Try n_patterns=2.

### Pattern time ranges seem wrong (very short windows)
**Possible causes**:
- Recording has sudden noise bursts (artifact or real)
- NMF components are noisy
- Increase `--n-patterns` to let K-means separate genuine patterns

### Results look inconsistent between runs
**Due to K-means randomness**. Options:
- Set random seed in config.yaml (advanced)
- Run multiple times to find consistent patterns
- Use n_patterns=2 for more stable results

---

## Next Steps

1. **Examine temporal results**: Review JSON output to understand file structure
2. **Validate interpretation**: Do time ranges match your domain knowledge?
3. **Extract events**: Use time ranges to segment audio for manual inspection
4. **Create labeled training set**: Use strong patterns for supervised classification
5. **Combine modes**: Use temporal clusters + file-level clusters for hierarchical analysis

---

## Summary

| Feature | Benefit |
|---------|---------|
| **Multiple patterns per file** | Discover sequential/overlapping vehicles |
| **Time-stamped results** | Know exactly when each pattern occurs |
| **Component tracking** | See which NMF bases activate together |
| **Strength metric** | Quantify pattern prominence |
| **JSON export** | Integrate with other tools |
| **Parallel processing** | 5x speedup on multi-core systems |

**Temporal clustering bridges unsupervised discovery (finding multiple patterns) and supervised learning (labeling those patterns for classification).**

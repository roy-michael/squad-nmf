# Component Activation Visualization Guide

**New Feature**: Visual representation of how different sound sources activate simultaneously or sequentially within audio files.

---

## Overview

When you run temporal clustering with the `--visualize` flag, the system generates **3 complementary visualizations** per audio file:

1. **Timeline Plot**: Component activations over time with cluster assignments
2. **Heatmap**: Intensity map showing when each component is active
3. **Summary Chart**: Mean activations compared across patterns

Together, these visualizations answer: **"How do different sound sources activate?"**

---

## Quick Start

### Generate Visualizations

```bash
python -m src temporal-cluster \
  --input-dir ./recordings \
  --n-patterns 2 \
  --visualize \
  --viz-dir ./visualizations
```

**Output**: `visualizations/<filename>_timeline.png`, `_heatmap.png`, `_summary.png`

---

## Visualization 1: Timeline Plot

### Purpose
Show all component activations over time with temporal cluster assignments.

### Three Panels

#### **Panel 1: Stacked Area Chart**
```
Axis Y: Activation Level (0 to max)
Axis X: Time (seconds)

Visual:
- Each color represents one NMF component
- Area height = how much that component contributes at each moment
- Stacked = all components sum to total spectrogram energy
```

**What it tells you:**
- **Spikes**: Strong, localized events (e.g., vehicle passing, transient)
- **Flat sections**: Consistent background noise
- **Color changes**: Different components dominating at different times

#### **Panel 2: Individual Component Lines**
```
Axis Y: Component activation (0 to ~0.5)
Axis X: Time (seconds)

Visual:
- Each component is a separate line
- Easier to see individual component behavior
- Line intersections = components switching dominance
```

**What it tells you:**
- **Sharp peaks**: Distinct acoustic events
- **Flat lines**: Inactive components
- **Overlapping lines**: Simultaneous activation (MULTIPLE VEHICLES!)

#### **Panel 3: Cluster Assignments**
```
Axis Y: Cluster ID
Axis X: Time (seconds)

Visual:
- Colored bands show temporal cluster membership
- Each time frame assigned to one cluster
- Band width = duration of that pattern
```

**What it tells you:**
- **Narrow bands**: Short-duration patterns (transients, echoes)
- **Wide bands**: Long-duration patterns (vehicles, ambient)
- **Band transitions**: When pattern changes occur
- **Multiple bands**: Multiple distinct sound sources

---

## Visualization 2: Heatmap

### Purpose
Show component activation intensity using color (red=active, black=inactive).

### How to Read

```
Axis Y: NMF components (Comp 0-5)
Axis X: Time (seconds)
Color: Red (0.30) -> Orange -> Yellow -> Dark -> Black (0.00)

Visual:
- Each pixel = one component at one time
- RED pixels = high activation
- BLACK pixels = no activation
- Vertical lines show when components activate
- Cyan dashed lines mark cluster boundaries
```

### Interpreting Patterns

#### **Pattern A: Single Vertical Red Line**
```
Heatmap shows:
  Comp 0: ██░░░░░░░░
  Comp 1: ░░░░░░░░░░
  Comp 2: ██░░░░░░░░
  Comp 3: ░░░░░░░░░░
  Comp 4: ░░░░░░░░░░
  Comp 5: ░░░░░░░░░░

Interpretation:
✓ Components 0 & 2 activate simultaneously
✓ Brief event (narrow red line)
✓ Single sound source or coherent event
```

#### **Pattern B: Two Separate Vertical Lines**
```
Heatmap shows:
  Comp 0: ██░░░░░░░░
  Comp 1: ░░░░░░░░░░
  Time:   ▲        ▲

Interpretation:
✓ Two distinct events (two vertical lines)
✓ Different components
✓ Sequential sources (different time windows)
```

#### **Pattern C: Overlapping Red Regions**
```
Heatmap shows:
  Comp 0: ████░░░░░░
  Comp 2: ░██████░░░
  Comp 4: ░░░░██████

Interpretation:
✓ Multiple components active at different times
✓ Sequence of related events
✓ Multiple sources passing through
```

#### **Pattern D: Simultaneous Wide Red Lines (SAME TIME!)**
```
Heatmap shows:
  Comp 0: ░░██░░░░░░
  Comp 2: ░░██░░░░░░  ← Same time window!
  Comp 4: ░░██░░░░░░

Interpretation:
✓✓ MULTIPLE COMPONENTS ACTIVE AT EXACT SAME TIME
✓✓ Multiple sound sources active simultaneously
✓✓ Mixed/overlapping acoustic signals
```

### Cyan Dashed Lines

The vertical cyan dashed lines mark **cluster boundaries** — where the temporal clustering algorithm decided a pattern change occurred.

```
← Different cluster → ← Different cluster →
```

These boundaries help confirm where pattern transitions happen.

---

## Visualization 3: Pattern Summary Chart

### Purpose
Compare mean component activations across temporal patterns.

### How to Read

```
Visual: Side-by-side bar charts
- One chart per temporal pattern
- X-axis: Component ID (0-5)
- Y-axis: Mean activation (0 to max)
- Bar colors: Component colors (same as timeline)
- RED BORDERS: Significant components (>0.05 activation)
```

### Interpreting Differences

#### **Pattern 0 vs Pattern 1**

**Pattern 0: All bars near zero**
```
Pattern 0 Mean Activations
  Comp 0: ░  (0.0001)
  Comp 1: ░  (0.0002)
  Comp 2: ░  (0.0001)
  Comp 3: ░  (0.0000)
  Comp 4: ░  (0.0000)
  Comp 5: ░  (0.0000)

Interpretation:
✓ Background/ambient noise
✓ Weak signal throughout
✓ No dominant components
```

**Pattern 1: Some bars have red borders**
```
Pattern 1 Mean Activations
  Comp 0: ████ (0.33) [RED BORDER]
  Comp 1: ██   (0.20) [RED BORDER]
  Comp 2: ░    (0.02)
  Comp 3: ░    (0.01)
  Comp 4: ░    (0.01)
  Comp 5: ░    (0.00)

Interpretation:
✓ Strong, distinct signal
✓ Components 0 & 1 dominate
✓ Likely a vehicle or specific sound type
```

### Comparing Patterns

**Completely Different Components?**
```
Pattern 0: Red borders on Comp [1,4]
Pattern 1: Red borders on Comp [0,2]
Pattern 2: Red borders on Comp [3,5]

→ Three different sound sources!
  Each uses different frequency (NMF basis)
```

**Same Components, Different Strength?**
```
Pattern 0: Comp 0 = 0.05
Pattern 1: Comp 0 = 0.25

→ Same source, different intensity
  (e.g., same vehicle accelerating)
```

**Multiple Patterns with Same Components?**
```
Pattern 0: Red borders on Comp [0,2]
Pattern 1: Red borders on Comp [0,2,3,4]

→ Pattern 1 is MORE COMPLEX
  Same source (Comp 0,2) + additional signal (Comp 3,4)
  Possibly two vehicles overlapping!
```

---

## Real Example: Simultaneous Vehicles

Let's analyze: **RBW6737_20250723_095000.wav**

### Timeline Analysis
```
Timeline Panel 2 (Individual Components):
- Most of file: All components flat (near 0)
- At 22.55 seconds: Component 0 SPIKES
- At 22.55 seconds: Component 5 SPIKES (SAME TIME!)
- At 22.93 seconds: Both drop back to zero

Interpretation:
✓ Background noise throughout (low activation)
✓ Sudden event at 22.55s
✓ Components 0 & 5 activate TOGETHER
✓ Event lasts ~0.38 seconds
✓ Likely a transient acoustic event
```

### Heatmap Analysis
```
Heatmap visualization:
- Mostly black (no activation)
- At 105.6 seconds (22.55s in frame units): RED VERTICAL LINE
- Components 0 & 5 light up RED
- Cyan dashed boundary shows cluster separation
- Pattern_1 (event) clearly separated from Pattern_0 (background)

Interpretation:
✓ Clear visual distinction between background and event
✓ Component 0 & 5 activate synchronously
✓ Multiple components needed to explain the event
```

### Summary Analysis
```
Pattern_0 (Background):
- All bars: ░ (near 0)
- No red borders
- Interpretation: Weak, unstructured background

Pattern_1 (Event):
- Component 0: ████ [RED BORDER]
- Component 5: ███ [RED BORDER]
- Component 1: ██ [RED BORDER]
- Component 2: █ (marginal)
- Others: ░

Interpretation:
✓ Strong, focused pattern
✓ Three components significantly activated
✓ Coherent acoustic signature
✓ Likely single event (one sound source)
```

---

## Detecting Simultaneous Sound Sources

### What to Look For

| Situation | Visual Indicator | Meaning |
|-----------|-----------------|---------|
| **Single vehicle** | One timeline spike, 1-2 components active | Sequential or isolated event |
| **Two vehicles overlapping** | Timeline spike with 3+ components active | Multiple sources need more components to explain |
| **Two vehicles sequential** | Two separate timeline spikes, different components | Different time windows, different sources |
| **Vehicle + background** | One spike + flat baseline, different components | Foreground signal + background |
| **Identical engines** | Single spike, same components both times | Indistinguishable sources (can't separate) |

### Critical Clue: Simultaneous Component Activation

If multiple components activate at the **exact same time**, you likely have:
1. **Multiple sound sources** overlapping, OR
2. **Complex waveform** from one source requiring multiple bases

Check the **summary chart**:
- If Pattern has 4+ red borders: Likely multiple sources
- If only 1-2 red borders: Likely single complex source

---

## Advanced: Reading the Math

### H Matrix Decomposition

What you're actually visualizing:

```
Mel Spectrogram ≈ W @ H

H = 6×15001 activation matrix
- 6 rows = 6 NMF basis functions
- 15001 columns = 15001 time frames (~2ms each at 8kHz)
- Each element = how much that basis contributes at that time

When you see "multiple components active at same time":
This means multiple H[component, time_frame] values are high

Which means multiple basis functions are contributing to 
the Mel spectrogram at that moment

Which means the sound is complex enough to need 
multiple frequency patterns simultaneously
```

### Why This Identifies Multiple Vehicles

```
Vehicle A:
- Propeller rotation: 50-200 Hz (Basis 1, 3)
- Cavitation noise: 3-8 kHz (Basis 2, 4)

Vehicle B:
- Different propeller: 100-300 Hz (Basis 0, 2)
- Different cavitation: 2-7 kHz (Basis 1, 5)

If both pass by simultaneously:
- H[0, t] high (Vehicle B's low frequency)
- H[1, t] high (Vehicle A's cavitation)
- H[2, t] high (Vehicle B's low frequency)
- H[3, t] high (Vehicle A's propeller)
- etc.

Result: 4+ components activated at time t
Visualization: Heatmap shows multiple red vertical lines at same x-position
Interpretation: Multiple sources active together!
```

---

## Usage Tips

### 1. Always Check All Three Plots

| Plot | Answers |
|------|---------|
| Timeline | When do components activate? (temporal structure) |
| Heatmap | Which exact moments have overlaps? (precise timing) |
| Summary | What are the key components per pattern? (identity) |

### 2. Combine with JSON Results

```bash
# JSON gives you numbers
cat temporal_analysis.json | grep "Pattern_1"
# Output: strength: 0.2121, components: [0, 2]

# Visualization confirms and shows
# Heatmap: Components 0 & 2 light up together ✓
# Summary: Red borders on exactly components 0 & 2 ✓
```

### 3. Use for Manual Validation

After looking at visualizations:
1. For Pattern_1 (vehicle): Extract 22.55-22.93s audio segment
2. Listen to the segment (should sound like single event)
3. Train classifier on this labeled segment
4. Use for supervised classification later

### 4. Automated Analysis Script

```python
import json
from pathlib import Path

with open('temporal_analysis.json') as f:
    results = json.load(f)

for filename, data in results.items():
    patterns = data['patterns']
    
    # Count patterns with overlapping times
    times = [(p['time_range'].split(' - ')) for p in patterns.values()]
    
    # Find patterns with many active components
    complex_patterns = [
        (name, p) for name, p in patterns.items()
        if len(p['active_components']) > 2
    ]
    
    if complex_patterns:
        print(f"{filename}: Complex patterns detected!")
        for name, pattern in complex_patterns:
            print(f"  {name}: components {pattern['active_components']}")
```

---

## Common Patterns You'll See

### Pattern 1: Ocean Ambient + Vehicle
```
Pattern_0: Flat baseline (0.001 strength)
Pattern_1: Spike with 2-3 active components (0.15 strength)

Visual cues:
- Timeline: Baseline + narrow spike
- Heatmap: Black + vertical red line
- Summary: Pattern_0 all zeros, Pattern_1 has peaks

→ Clear foreground/background separation
```

### Pattern 2: Multiple Vehicles
```
Pattern_0: Spike 1 (0.12 strength, Comp 2,3)
Pattern_1: Spike 2 (0.10 strength, Comp 0,4)
Pattern_2: Overlap (0.18 strength, Comp 0,2,3,4)

Visual cues:
- Timeline: Multiple spikes with different components
- Heatmap: Multiple vertical red lines at different positions + overlap region
- Summary: Each pattern has different red borders

→ Multiple distinct vehicles identified
```

### Pattern 3: Same Vehicle, Different Modes
```
Pattern_0: Spike (0.10 strength, Comp 1,2)
Pattern_1: Spike (0.20 strength, Comp 1,2)

Visual cues:
- Timeline: Same components activated, different heights
- Heatmap: Red lines at same positions but different intensities
- Summary: Red borders on same components, different bar heights

→ Same source accelerating or changing mode
```

---

## Troubleshooting Visualizations

### "All patterns look the same"
- Try decreasing `--n-patterns` (e.g., 2 instead of 3)
- Recording may have uniform background noise
- Use heatmap to confirm all components have similar activation

### "Heatmap is all black"
- Audio file is very quiet or silence
- Check original audio file is not corrupted
- Try with different input directory

### "Too many small patterns"
- `--n-patterns` value too high
- K-means is over-clustering noise variations
- Reduce to 2-3 patterns and re-run

### "One huge pattern, one tiny pattern"
- Common! One pattern is background, others are foreground
- This is expected and correct
- Use summary chart to see strength differences

---

## Summary: Reading the Visualizations

**Timeline**: Shows when and which components are active  
**Heatmap**: Shows exact simultaneous activation (red pixels at same X coordinate)  
**Summary**: Shows which components define each pattern  

**For simultaneous sources**: Look for multiple components lighting up at the same time in both heatmap and summary.

**For sequential sources**: Look for different time windows in timeline and different components in heatmap.

**For mixed signals**: Look for high component count (3+) and high strength metric.

---

## Next Steps

1. Generate visualizations for your recordings
2. Open the PNG files in an image viewer
3. Identify patterns matching your domain knowledge
4. Use time ranges from JSON to extract audio segments
5. Manually label segments
6. Train supervised classifier on labeled data

See `TEMPORAL_CLUSTERING_GUIDE.md` for full workflow.

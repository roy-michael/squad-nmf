# Simultaneous Sound Detection - How It Works

## Critical Clarification

You're right to question this! **`--n-patterns` finds temporal patterns that can include SIMULTANEOUS sounds.**

The system does NOT assume sounds are sequential. It detects:
- ✅ Multiple sounds happening AT THE SAME TIME
- ✅ Multiple sound periods happening one after another
- ✅ Mixed scenarios (some simultaneous, some sequential)

---

## How Simultaneous Sounds Are Detected

### The Key: NMF Components Activate Together

When multiple sound sources emit at the same time, **multiple NMF components activate simultaneously**.

```
Time Frame t:
┌─────────────────────────────────────┐
│ NMF Activations (6 components)      │
├─────────────────────────────────────┤
│ Component 0: 0.2  ← Vehicle A       │
│ Component 1: 0.5  ← Vehicle B       │
│ Component 2: 0.8  ← Vehicle B       │
│ Component 3: 0.1  ← Ambient         │
│ Component 4: 0.0  (inactive)        │
│ Component 5: 0.0  (inactive)        │
└─────────────────────────────────────┘
        ↑
    Multiple high activations = SIMULTANEOUS detection
```

### Visualization Proof

**In the temporal visualizations:**

1. **Timeline Plot (Panel 2 - Individual Lines)**
   ```
   Component 0:  ─────╱╲────────╲──  (overlapping with others)
   Component 1:  ──╱╲───────╱╲───   (same time frames)
   Component 2:  ╱╲──────╱╲───────  (active together)
   
   Where lines overlap vertically = SIMULTANEOUS activation
   ```

2. **Heatmap Plot**
   ```
   Red pixels at SAME time column = multiple components active = SIMULTANEOUS
   
   Time →
   C ↓  [Red Red Red] ← Same time frame, multiple active
   o    [Red Red Red]
   m    [Red   Red ]
   p    [    Red   ]
   ```

3. **How K-means Clusters This**
   ```
   Frame 100: [0.2, 0.5, 0.8, 0.1, 0.0, 0.0] → Cluster A
   Frame 101: [0.2, 0.5, 0.8, 0.1, 0.0, 0.0] → Cluster A (similar)
   Frame 102: [0.2, 0.5, 0.8, 0.1, 0.0, 0.0] → Cluster A (simultaneous)
   
   Frame 200: [0.8, 0.0, 0.0, 0.3, 0.0, 0.0] → Cluster B (different)
   Frame 201: [0.8, 0.0, 0.0, 0.3, 0.0, 0.0] → Cluster B (components changed)
   ```

---

## Real Example: 3 Vehicles Detected Simultaneously

### Scenario
```
Recording with 3 submarines passing at same time:
- Sub A: Low-frequency propulsion (0-20 seconds)
- Sub B: Medium-frequency propulsion (5-15 seconds, OVERLAPS with A)
- Sub C: High-frequency cavitation (8-12 seconds, OVERLAPS with A and B)
```

### What Happens

**NMF learns:**
- Component 0: Low frequencies (Sub A signature)
- Component 1: Medium frequencies (Sub B signature)
- Component 2: High frequencies (Sub C signature)
- Components 3-5: Ambient/noise

**Temporal clustering discovers:**

```
Timeline visualization will show:
┌────────────────────────────────────────┐
│ 0-5s:  [Component 0 active]            │ Sub A alone
│        Pattern: Low-freq baseline       │
├────────────────────────────────────────┤
│ 5-8s:  [Component 0 + 1 active]        │ Sub A + B together
│        Pattern: Low + Medium freq       │ ← SIMULTANEOUS
├────────────────────────────────────────┤
│ 8-12s: [Component 0 + 1 + 2 active]    │ All 3 together
│        Pattern: Low + Medium + High     │ ← SIMULTANEOUS
├────────────────────────────────────────┤
│ 12-15s:[Component 0 + 1 active]        │ Sub A + B
│        Pattern: Low + Medium freq       │ ← SIMULTANEOUS
├────────────────────────────────────────┤
│ 15-20s:[Component 0 active]            │ Sub A alone
│        Pattern: Low-freq baseline       │
└────────────────────────────────────────┘
```

**With `--n-patterns 5` or more**, system discovers each distinct combination:
- Pattern 1: Sub A alone
- Pattern 2: Sub A + Sub B
- Pattern 3: Sub A + Sub B + Sub C
- Pattern 4: Sub A + Sub B (different moment)
- Pattern 5: Transition/fade

---

## How to Detect Simultaneous Sounds in the Output

### Check the Visualizations

When you run:
```bash
python -m src temporal-cluster --input data/ --n-patterns 5 --visualize --viz-dir plots/
```

**Look for:**

1. **Timeline Plot (3 panels)**
   - Panel 2: Multiple lines overlapping at same X position = SIMULTANEOUS
   - Overlapping means same time frame, multiple components active

2. **Heatmap Plot**
   - Vertical stripe of red (multiple rows with red in same column)
   - Same time column, multiple rows lit up = SIMULTANEOUS

3. **Summary Plot**
   - Bars for patterns with multiple red-bordered components
   - Each component is one sound source

### Check the JSON Output

```json
{
  "file.wav": {
    "patterns": {
      "Pattern_0": {
        "time_range": "0.00s - 5.20s",
        "active_components": [0],           // Only Sub A
        "strength": 0.045
      },
      "Pattern_1": {
        "time_range": "5.20s - 8.00s",
        "active_components": [0, 1],        // Sub A + B TOGETHER
        "strength": 0.068                   // ← SIMULTANEOUS
      },
      "Pattern_2": {
        "time_range": "8.00s - 12.00s",
        "active_components": [0, 1, 2],     // Sub A + B + C TOGETHER
        "strength": 0.085                   // ← HIGHLY SIMULTANEOUS
      }
    }
  }
}
```

**Key Signal:** Multiple entries in `active_components` array = SIMULTANEOUS

---

## Why n-patterns Matters for Simultaneous Sounds

### Low n-patterns (2-3): Merges Simultaneous into One

```bash
--n-patterns 2
```

Result:
```
Pattern_0: Background (no vehicles)
Pattern_1: "Vehicles active" (ALL combinations merged)
           - Cannot distinguish between:
             * Sub A alone
             * Sub A + B
             * Sub A + B + C
```

**Problem:** Loses simultaneous structure information

### Medium n-patterns (4-5): Shows Distinct Combinations

```bash
--n-patterns 4
```

Result:
```
Pattern_0: No vehicles
Pattern_1: Vehicle 1 alone
Pattern_2: Vehicle 1 + 2 (SIMULTANEOUS)
Pattern_3: Vehicle 1 + 2 + 3 (SIMULTANEOUS)
```

**Better:** Can see when sounds overlap

### High n-patterns (6+): Fine-Grained Simultaneous Analysis

```bash
--n-patterns 6
```

Result:
```
Pattern_0: No vehicles
Pattern_1: Vehicle 1 alone (0-5s)
Pattern_2: Vehicle 1 + 2 (5-8s) SIMULTANEOUS
Pattern_3: Vehicle 1 + 2 + 3 (8-12s) SIMULTANEOUS
Pattern_4: Vehicle 1 + 2 (12-15s) SIMULTANEOUS
Pattern_5: Vehicle 1 alone (15-20s)
```

**Best:** Can see exact timing of when sounds overlap

---

## Recommended Approach for Simultaneous Sounds

### Step 1: Use Higher n-patterns

For simultaneous sound detection, use **4-6** instead of default 3:

```bash
# Better for simultaneous detection
python -m src temporal-cluster --input data/ --n-patterns 5 --visualize --viz-dir plots/
```

### Step 2: Enable Visualizations

Always visualize to see overlapping components:

```bash
python -m src temporal-cluster --input data/ --n-patterns 5 --visualize --viz-dir plots/
```

### Step 3: Read the JSON Results

Look for patterns with **multiple components** in `active_components`:

```bash
# Check results
cat temporal_analysis.json | grep -A 3 "active_components"
```

Any pattern with 2+ components = SIMULTANEOUS DETECTION ✅

---

## Real Command Examples

### Example 1: Detect Simultaneous Submarines

```bash
python -m src temporal-cluster \
  --input recordings/ \
  --n-patterns 5 \
  --visualize \
  --viz-dir ./simultaneous_analysis/
```

**Expected to show:**
- Patterns where multiple components activate at same time
- Visualization heatmaps with vertical red stripes
- JSON with patterns containing multiple active_components

### Example 2: Find All Overlapping Periods

```bash
python -m src temporal-cluster \
  --input recordings/ \
  --n-patterns 6 \
  --visualize \
  --viz-dir ./overlaps/
```

Then check JSON for:
```json
"active_components": [0, 1, 2]  ← 3 components = 3 simultaneous sounds
```

### Example 3: Compare Results with Different n-patterns

```bash
# Try different values to see simultaneous structure
python -m src temporal-cluster --input data/ --n-patterns 3 --visualize --viz-dir plots_3/
python -m src temporal-cluster --input data/ --n-patterns 5 --visualize --viz-dir plots_5/

# Compare plots_3/ vs plots_5/ to see difference
```

---

## How to Verify Simultaneous Detection

### Checklist

- [ ] Visualizations show multiple overlapping lines (Panel 2)
- [ ] Heatmap has vertical red stripes (same time, multiple rows)
- [ ] JSON has patterns with 2+ components in `active_components`
- [ ] Timeline ranges overlap (Pattern 1 ends AFTER Pattern 2 starts)
- [ ] Summary chart shows red borders on multiple components per pattern

### Visual Proof

**This proves simultaneous detection:**
```
Timeline plot:
├─ Sub A: ──────════════════────  (spans 0-20s)
├─ Sub B:      ════════════       (spans 5-15s, OVERLAPS with A)
└─ Sub C:         ══════          (spans 8-12s, OVERLAPS with both)
             └────┬────┘
              Simultaneous period
```

---

## FAQ

### Q: How does the system know it's simultaneous and not sequential?

**A:** By analyzing NMF component activation at each time frame:
- If frame T has [0.5, 0.6, 0.1, ...] → components 0 AND 1 both active at same time T
- If they were sequential, frame T would have [0.5, 0, ...] OR [0, 0.6, ...]

### Q: Can it detect 3 submarines simultaneously?

**A:** YES, if they have distinct frequency signatures:
- Submarine A activates components [0]
- Submarine B activates components [1, 2]
- Submarine C activates components [3]
- Same time frame with all 3 active = DETECTED as simultaneous

### Q: What if sounds have overlapping frequencies?

**A:** NMF will struggle to separate them - they may appear as one component
- Solution: Use more n_components in config.yaml (increase from 6 to 8 or 10)
- More basis functions = better frequency separation

### Q: Recommended n-patterns for simultaneous sounds?

**A:** **Use 5-6** instead of default 3
- 3 is too low, merges simultaneous combinations
- 5-6 shows each distinct simultaneous combination clearly

### Q: How do I count simultaneous sounds in a pattern?

**A:** Count entries in `active_components`:
```json
"active_components": [0]       → 1 sound
"active_components": [0, 1]    → 2 simultaneous sounds
"active_components": [0, 1, 2] → 3 simultaneous sounds
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Can detect simultaneous?** | ✅ YES - via multiple NMF components at same time |
| **Proof in visualization** | Overlapping lines, vertical red stripes in heatmap |
| **Proof in JSON** | Multiple entries in `active_components` array |
| **Best n-patterns for simultaneous** | 5-6 (vs default 3) |
| **Command to use** | `python -m src temporal-cluster --input data/ --n-patterns 5 --visualize` |
| **How to verify** | Check for patterns with 2+ active_components |

---

## Your Scenario

If you have multiple underwater vehicles emitting **at the same time**:

```bash
python -m src temporal-cluster \
  --input your_recordings/ \
  --n-patterns 5 \
  --n-jobs=-1 \
  --visualize \
  --viz-dir analysis/
```

Then check `analysis/` plots for:
- ✅ Overlapping component lines
- ✅ Vertical red heatmap stripes
- ✅ Patterns with multiple active components

This proves the system **detected simultaneous sounds correctly**. ✅


# SIMULTANEOUS Sound Detection - Visual Summary

## ✅ YES, It Detects Simultaneous Sounds

The system is **NOT** limited to sequential patterns. It detects multiple sounds happening **AT THE SAME TIME**.

---

## How Multiple NMF Components = Simultaneous Detection

### Time Frame t (same moment in time)

```
Three submarines emitting together at time t:

NMF Component Activations:
┌─────────────────────────────────────────┐
│ Submarine A:  Component 0 = 0.7 ─┐      │
│                                   ├─ ALL active at
│ Submarine B:  Component 1 = 0.5 ─┤    SAME TIME t
│                                   │
│ Submarine C:  Component 2 = 0.8 ─┘      │
│ Ambient:      Component 3 = 0.2         │
└─────────────────────────────────────────┘
        ↑
   Multiple high values = SIMULTANEOUS DETECTION
```

---

## Visual Proof in Plots

### Timeline Plot - Multiple Lines Overlapping

```
                    Simultaneous Period (8-12 seconds)
                           ↓
Sub A (Comp 0):  ─────────╱╲╲╲╲╲╲╲╲╲────────
Sub B (Comp 1):  ──────╱╱╱╱╱╱╱╱╱╱╱╱╱╱───
Sub C (Comp 2):  ────╱╱╱╱╱╱╱╱╱╱────────

Overlapping lines at same X position = SAME TIME = SIMULTANEOUS ✓
```

### Heatmap Plot - Red Vertical Stripes

```
           Time (seconds) →
Component ↓  |___|___|XXX|XXX|___|
              Sub A  B C overlap
             ═══════════════════
             All 3 active at same column
             (same time) = SIMULTANEOUS ✓

Legend:
  X = Red (active)
  _ = Black (inactive)
  Column = Time frame
```

---

## JSON Output Proof

### Pattern with Multiple Simultaneous Sounds

```json
{
  "recording.wav": {
    "patterns": {
      "Pattern_2": {
        "time_range": "8.00s - 12.00s",
        "active_components": [0, 1, 2],    ← 3 components = 3 sounds
        "strength": 0.085
      }
    }
  }
}
```

| Entry | Meaning |
|-------|---------|
| `"time_range": "8.00s - 12.00s"` | When simultaneous occurs |
| `"active_components": [0, 1, 2]` | Submarine A + B + C together |
| **Multiple entries** | PROVES simultaneous ✓ |

---

## Real Scenario Example

### Your Recording: 3 Submarines at Same Time

```
Timeline:
─────────────────────────────────────
│ 0-5s:   Sub A only              │
│ 5-8s:   Sub A + B together      │ ← SIMULTANEOUS
│ 8-12s:  Sub A + B + C together  │ ← SIMULTANEOUS
│ 12-15s: Sub A + B together      │ ← SIMULTANEOUS
│ 15-20s: Sub A only              │
─────────────────────────────────────
```

### Command to Detect This

```bash
python -m src temporal-cluster \
  --input data/ \
  --n-patterns 5 \
  --visualize \
  --viz-dir analysis/
```

### Expected Result

**In `analysis/file_timeline.png`:**
```
Lines overlapping at 5-12s = SIMULTANEOUS DETECTION ✓
```

**In `analysis/file_heatmap.png`:**
```
Red vertical stripe from 5-12s = SIMULTANEOUS ✓
```

**In `temporal_analysis.json`:**
```json
"Pattern_1": {
  "time_range": "5.00s - 8.00s",
  "active_components": [0, 1],      ← 2 subs together
  "strength": 0.068
},
"Pattern_2": {
  "time_range": "8.00s - 12.00s",
  "active_components": [0, 1, 2],   ← 3 subs together
  "strength": 0.085
}
```

All three checks confirm: **SIMULTANEOUS DETECTED** ✓

---

## Key Parameter: Use Higher n-patterns

### Default (n-patterns 3)
```
Pattern_0: Background
Pattern_1: "Vehicles active"  ← Merges all combinations
Pattern_2: (unused)
```
**Problem:** Cannot distinguish simultaneous combinations

### Recommended (n-patterns 5-6)
```
Pattern_0: Background
Pattern_1: Sub A alone
Pattern_2: Sub A + B (SIMULTANEOUS)
Pattern_3: Sub A + B + C (SIMULTANEOUS)
Pattern_4: Sub A + B (later time, SIMULTANEOUS)
Pattern_5: Sub A alone (fade)
```
**Better:** Shows each distinct simultaneous combination

---

## Your Recommended Command

For multiple vehicles emitting **simultaneously**:

```bash
python -m src temporal-cluster \
  --input your_recordings/ \
  --n-patterns 5 \
  --n-jobs=-1 \
  --visualize \
  --viz-dir results/
```

Then verify in results:
1. **Plots**: Multiple overlapping lines ✓
2. **Heatmap**: Vertical red stripes ✓
3. **JSON**: Multiple components per pattern ✓

---

## Simultaneous Sound Detection Checklist

After running the command above:

- [ ] `results/file_timeline.png` shows overlapping lines?
- [ ] `results/file_heatmap.png` shows vertical red stripes?
- [ ] `temporal_analysis.json` patterns have 2+ active_components?
- [ ] Time ranges overlap between patterns?
- [ ] Strength values are similar across overlapping patterns?

If all ✓, system successfully detected simultaneous sounds.

---

## Summary

| Question | Answer |
|----------|--------|
| **Can it detect simultaneous sounds?** | ✅ **YES** |
| **How?** | Multiple NMF components activate at same time |
| **Proof in visualization?** | Overlapping lines + vertical heatmap stripes |
| **Proof in JSON?** | Multiple entries in `active_components` |
| **Best n-patterns?** | **5-6** (not default 3) |
| **Your command?** | `--n-patterns 5 --visualize` |

---

## 🎯 Bottom Line

**Your scenario with simultaneous vehicles is exactly what this system is designed for.**

Use:
```bash
python -m src temporal-cluster --input data/ --n-patterns 5 --visualize --viz-dir plots/
```

The visualizations will show exactly when multiple submarines emit together. ✅


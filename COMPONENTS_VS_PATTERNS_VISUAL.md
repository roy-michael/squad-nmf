# n_components vs n-patterns - Quick Visual

## One-Sentence Summary

**n_components** = Frequency dimension (WHAT sounds exist)
**n-patterns** = Time dimension (WHEN they change)

---

## Visual Comparison

### n_components = Fixed "Dictionary"

```
┌─────────────────────────────────────┐
│  NMF COMPONENT DICTIONARY           │
│  (Set in config.yaml)               │
├─────────────────────────────────────┤
│                                     │
│  Component 0: Low frequencies       │
│  ════════════════════════════       │
│  "Submarine propulsion"             │
│                                     │
│  Component 1: Medium frequencies    │
│  ════════════════════════════       │
│  "Cavitation noise"                 │
│                                     │
│  Component 2: High frequencies      │
│  ════════════════════════════       │
│  "Machinery"                        │
│                                     │
│  Components 3-5: Other patterns     │
│  ════════════════════════════       │
│                                     │
└─────────────────────────────────────┘

Used for ALL recordings
Set once, never changes (unless retraining)
```

### n-patterns = Variable "Combinations"

```
File 1: two_subs.wav
┌──────────────────────────────────┐
│  TEMPORAL PATTERNS (n-patterns)  │
│  (Set on command line)           │
├──────────────────────────────────┤
│ Pattern 0 (0-10s): [Comp 0]      │
│ Pattern 1 (10-20s): [Comp 0, 1]  │
│ Pattern 2 (20-30s): [Comp 0]     │
└──────────────────────────────────┘

File 2: ship.wav
┌──────────────────────────────────┐
│  TEMPORAL PATTERNS (n-patterns)  │
├──────────────────────────────────┤
│ Pattern 0 (0-15s): [Comp 2]      │
│ Pattern 1 (15-30s): [Comp 2, 3]  │
│ Pattern 2 (unused)               │
└──────────────────────────────────┘

Different for EACH recording
Set per analysis, changed frequently
```

---

## Timeline Example

### Recording: Two Submarines

```
Time →   0s        10s        20s        30s
         ├──────────┼──────────┼──────────┤

NMF
Components:
(Fixed)
  Comp 0: █████████████████████
          (Sub A throughout)
  
  Comp 1:           █████████████
                    (Sub B joins)
  
  Comp 2:                    ██████████
                            (Cavitation)

K-means
Temporal
Patterns:
(Per-file)
  Pat 0:  [Comp 0 alone]
  Pat 1:  [Comp 0 + 1 together]  ← SIMULTANEOUS
  Pat 2:  [Comp 1 + 2 together]
```

**Components** show WHAT frequencies are active
**Patterns** show WHEN they activate

---

## Configuration vs Command Line

### Setting n_components (Global)

```
config.yaml:
───────────────
nmf:
  n_components: 6

Applied to ALL files during preprocessing
```

### Setting n-patterns (Per-analysis)

```
Command line:
─────────────
python -m src temporal-cluster --input data/ --n-patterns 3

Applied per-file during temporal analysis
```

---

## Changing Each

### Changing n_components

```
Current: n_components = 6

Need more detail?
  1. Edit config.yaml → n_components: 8
  2. Retrain model → python -m src train ...
  3. Re-classify files
  
Time cost: 5-10 minutes (retraining required)
```

### Changing n-patterns

```
Current: n-patterns = 3

Need more time periods?
  1. Run command → --n-patterns 5
  
Time cost: <1 second (no retraining)
```

---

## Analogy: Cooking a Dish

### n_components = Pantry (Fixed)

```
You have 6 ingredients:
  1. Salt
  2. Pepper
  3. Garlic
  4. Olive Oil
  5. Lemon
  6. Herbs

These ingredients are FIXED
(stocked once, used for all recipes)
```

### n-patterns = Recipe Steps (Variable)

```
Recipe for "Dish 1":
  Step 1 (0-5min): Add salt + pepper
  Step 2 (5-10min): Add garlic + olive oil
  Step 3 (10-15min): Add lemon + herbs

Recipe for "Dish 2":
  Step 1 (0-10min): Add salt + garlic
  Step 2 (10-20min): Add pepper + olive oil

Same ingredients, different combinations at different times
```

---

## Where Each Affects Output

### n_components Affects

```
What frequencies are extracted:
└─ 6 components × time frames
   Affects quality of features
   Affects classification accuracy
   Affects how well we understand each submarine
```

### n-patterns Affects

```
How time is segmented:
└─ 3 temporal clusters
   Affects granularity of temporal breakdown
   Affects visibility of simultaneous periods
   Affects understanding of mixing
```

---

## Decision Matrix

### When to Adjust n_components

```
Observation: Features are too coarse
             Classification accuracy is low
             
Action: Increase n_components in config.yaml
        Retrain model
        
Cost: Takes time, affects all files
```

### When to Adjust n-patterns

```
Observation: Missing temporal detail
             Cannot see simultaneous periods
             
Action: Increase --n-patterns on command line
        Re-analyze (no retraining)
        
Cost: Fast, per-file, no retraining
```

---

## Summary

| Dimension | n_components | n-patterns |
|-----------|--------------|-----------|
| **Controls** | FREQUENCY | TIME |
| **Location** | config.yaml | Command line |
| **Scope** | Global | Per-file |
| **Cost to change** | High (retrain) | Low (re-analyze) |
| **Typical range** | 4-12 | 2-6 |
| **Question** | "What sounds?" | "When do they mix?" |

---

## Your Scenario

### Setup (config.yaml)
```yaml
nmf:
  n_components: 6    # Learn 6 frequency signatures
```

### Command (terminal)
```bash
python -m src temporal-cluster --input data/ --n-patterns 5
                                            # Temporal detail
```

### Result
```
6 frequencies + 5 time periods 
= Complete picture of submarines mixing
```

✅ **Both needed, different jobs, different dimensions.**


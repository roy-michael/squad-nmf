# n-patterns vs n_components - Key Differences

## Quick Answer

**NO, they are NOT the same!**

| Parameter | Dimension | Purpose | Example |
|-----------|-----------|---------|---------|
| **n_components** | **Frequency** | How many acoustic building blocks to learn | "Find 6 different frequency signatures in ALL recordings" |
| **n-patterns** | **Time** | How many distinct time periods to find in EACH file | "Find 3 different temporal states within THIS recording" |

They control completely different aspects of the analysis.

---

## The Analogy: Cooking

### n_components = Ingredients

```
Pantry (learning phase):
├─ Salt (Component 0: frequency pattern A)
├─ Pepper (Component 1: frequency pattern B)
├─ Garlic (Component 2: frequency pattern C)
├─ Olive Oil (Component 3: frequency pattern D)
├─ Lemon (Component 4: frequency pattern E)
└─ Herbs (Component 5: frequency pattern F)

Fixed set of 6 ingredients for ALL recipes
```

### n-patterns = Recipes

```
Specific Dish (time period clustering):
├─ Recipe 1: Salt + Pepper (0-5 seconds)
├─ Recipe 2: Garlic + Olive Oil (5-10 seconds)
└─ Recipe 3: Lemon + Herbs (10-15 seconds)

Uses the 6 ingredients, but combines them differently at different times
```

---

## Technical Explanation

### n_components (Frequency Dimension)

```
LEARNING PHASE (Training)
║
║  Input: All 100 audio files
║  ├─ File 1: Submarine A
║  ├─ File 2: Submarine B
║  ├─ File 3: Ship
║  └─ File 4: Whale (mistaken for ship)
║
→ NMF learns: n_components = 6
  ├─ Component 0: Low-frequency propulsion (subs)
  ├─ Component 1: Medium-frequency cavitation
  ├─ Component 2: High-frequency machinery
  ├─ Component 3: Very low-frequency hull resonance
  ├─ Component 4: Ocean ambient noise
  └─ Component 5: Electrical noise
  
These 6 components are FIXED for all analysis
They represent "what frequency patterns exist"
```

### n-patterns (Time Dimension)

```
ANALYSIS PHASE (After learning n_components)
║
║  Input: Single file "multi_vehicle_recording.wav"
║  Duration: 30 seconds
║
→ K-means clusters time frames: n-patterns = 3
  ├─ Pattern 0: (0-10s) Component 0 dominant
  │             Uses Component 0 from the 6 available
  │
  ├─ Pattern 1: (10-20s) Components 0 + 1 dominant
  │             Uses Components 0, 1 from the 6 available
  │
  └─ Pattern 2: (20-30s) Components 0 + 1 + 2 dominant
                Uses Components 0, 1, 2 from the 6 available

These 3 patterns are SPECIFIC to this file
They represent "how does the mixture change over time"
```

---

## Concrete Example: 2 Submarines

### Stage 1: Learn Frequency Patterns (n_components)

```bash
# In config.yaml:
nmf:
  n_components: 6    ← Learn 6 frequency signatures from training data

# Applied to all recordings:
# Training data: 100 diverse underwater files
# Result: NMF learns
#   Component 0: Sub A's typical low frequency
#   Component 1: Sub B's medium frequency
#   Component 2: Cavitation noise
#   Components 3-5: Ambient, machinery, electrical
```

**These 6 components are GLOBAL** — same for all files.

### Stage 2: Analyze Temporal Patterns (n-patterns)

```bash
# Command:
python -m src temporal-cluster --input data/ --n-patterns 3

# Applied to EACH file separately:
# File: two_subs_together.wav (30 seconds)
# K-means finds: n-patterns = 3
#
# Pattern 0 (0-10s): Component 0 active
#                    (Sub A alone)
#
# Pattern 1 (10-20s): Components 0 + 1 active
#                     (Sub A + Sub B together)
#
# Pattern 2 (20-30s): Component 0 active
#                     (Sub A alone again)
```

**These 3 patterns are LOCAL** — specific to this one file.

---

## Visual Difference

### n_components: Fixed Dictionary (ALL FILES)

```
┌─────────────────────────────────┐
│     NMF COMPONENT DICTIONARY    │
│  (Learned once, used for all)   │
├─────────────────────────────────┤
│ Component 0: [freq 1, 2, 3, ...]│
│ Component 1: [freq 4, 5, 6, ...]│
│ Component 2: [freq 7, 8, 9, ...]│
│ Component 3: [freq 10, 11, ...]│
│ Component 4: [freq 12, 13, ...]│
│ Component 5: [freq 14, 15, ...]│
└─────────────────────────────────┘
         ↓ Applied to
   ┌──────────────────────────────────┐
   │    File 1: sub_a_passing.wav     │
   │    File 2: sub_b_passing.wav     │
   │    File 3: two_subs_together.wav │
   │    File 4: ship_engine.wav       │
   └──────────────────────────────────┘
```

### n-patterns: Temporal Breakdown (EACH FILE)

```
File: two_subs_together.wav
├─ Pattern 0: (0-10s) Uses [Comp 0]
├─ Pattern 1: (10-20s) Uses [Comp 0, 1]
└─ Pattern 2: (20-30s) Uses [Comp 0]

File: ship_engine.wav
├─ Pattern 0: (0-15s) Uses [Comp 2]
├─ Pattern 1: (15-30s) Uses [Comp 2, 3]
└─ Pattern 2: (unused)

Each file has DIFFERENT patterns based on its time evolution
```

---

## Why We Need Both

### Problem with ONLY n_components

```bash
# If we only had n_components = 6
What we can see:
  ✓ "This file contains low-freq (Comp 0) and mid-freq (Comp 1)"
  ✗ "When did the transition happen?"
  ✗ "Are they simultaneous or sequential?"
  ✗ "How many distinct time periods?"

Result: No temporal information, file-level classification only
```

### Problem with ONLY n-patterns

```bash
# If we only had n-patterns = 3
What we can see:
  ✓ "Pattern 0, Pattern 1, Pattern 2 exist"
  ✗ "What frequency content defines each pattern?"
  ✗ "Which NMF components are active?"
  ✗ "Can we compare to other files?"

Result: No frequency information, patterns are meaningless
```

### Solution: BOTH TOGETHER

```bash
# With n_components = 6 AND n-patterns = 3
What we can see:
  ✓ Pattern 0 (0-10s): Dominated by Component 0 (low freq)
  ✓ Pattern 1 (10-20s): Dominated by Components 0 + 1 (low + mid)
  ✓ Pattern 2 (20-30s): Dominated by Component 0 (low freq)
  ✓ Comparison: File 1's patterns vs File 2's patterns
  ✓ Frequency interpretation: "Component 0 is submarine propulsion"

Result: Complete temporal AND frequency understanding
```

---

## Real Example: Analyzing a Recording

### Input File: multi_vehicle.wav (30 seconds)

### Step 1: Apply n_components (Global, Pre-computed)

```
NMF Extracts (6 components × 3000 time frames):

Time →  0.0s              15.0s              30.0s
        ├──────────────────┼──────────────────┤
Comp 0: ████████████████████░░░░░░░░░░░░░░░░░░  (submarine A)
Comp 1: ░░░░░░░░░░░░░░░░░░████████████░░░░░░░░  (submarine B)
Comp 2: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████  (cavitation)
Comp 3: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (unused)
Comp 4: ████░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░  (ambient)
Comp 5: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (unused)

Where █ = component active, ░ = component inactive
```

### Step 2: Apply n-patterns (Local, Per-File)

```
K-means groups similar time frames:

Pattern 0 (0-10s):    [Comp 0 active]
                      "Submarine A alone"

Pattern 1 (10-20s):   [Comp 0 + 1 active]
                      "Submarine A + B together"

Pattern 2 (20-30s):   [Comp 1 + 2 active]
                      "Submarine B + cavitation"

Each pattern is a DIFFERENT TIME PERIOD
Each uses a DIFFERENT COMBINATION of the 6 fixed components
```

---

## Configuration Example

### config.yaml (n_components - GLOBAL)

```yaml
nmf:
  n_components: 6          ← Frequency dimension
                           ← Learned from training data
                           ← Same for ALL files
```

### Command Line (n-patterns - LOCAL)

```bash
python -m src temporal-cluster \
  --input data/ \
  --n-patterns 3            ← Time dimension
                            ← Applied per file
                            ← Can change per analysis
```

---

## Why They're Different

### n_components Controls

- **What** gets learned: "Learn 6 frequency signatures from training data"
- **Scope**: Global (entire training dataset)
- **When**: During preprocessing (once)
- **Change frequency**: Rarely (affects training)
- **Values**: Typically 4-12

### n-patterns Controls

- **What** gets discovered: "Find 3 time periods with different activations"
- **Scope**: Local (each file independently)
- **When**: During analysis (every run)
- **Change frequency**: Often (experiment per analysis)
- **Values**: Typically 2-6

---

## When to Adjust Each

### Adjust n_components When:

```
Symptoms:
  ✗ Features are too coarse (n_components=3, not enough detail)
  ✗ Features are too fine (n_components=12, noisy/fragmented)
  ✗ Classification accuracy is poor

Fix:
  # Edit config.yaml
  nmf:
    n_components: 8    # Increase from 6
  
  # Retrain model
  python -m src train --data training_data/ --output model.pkl
```

### Adjust n-patterns When:

```
Symptoms:
  ✗ Temporal patterns too coarse (missing detail)
  ✗ Temporal patterns too fine (too fragmented)
  ✗ Can't see simultaneous periods

Fix:
  # Just change command-line parameter
  python -m src temporal-cluster --input data/ --n-patterns 5
  # No retraining needed!
```

---

## Analogy: Photography

### n_components = Camera Lenses

```
Camera Kit (once per trip):
├─ Wide-angle lens (learns broad patterns)
├─ Standard lens (learns typical patterns)
├─ Telephoto lens (learns detailed patterns)
└─ Macro lens (learns fine patterns)

Decision: "Which lenses to bring?"
Answer: "6 different focal lengths"

Used for ALL photos on this trip
```

### n-patterns = Photo Framing

```
Single Photo (specific moment):
├─ Wide shot (whole scene)
├─ Medium shot (focus on subject)
└─ Close-up (fine details)

Decision: "How many frames to take?"
Answer: "3 different compositions"

Different for each photo
```

---

## Summary Table

| Aspect | n_components | n-patterns |
|--------|-------------|-----------|
| **Dimension** | Frequency | Time |
| **Purpose** | Learn acoustic basis functions | Discover time periods |
| **Scope** | Global (all files) | Local (per file) |
| **Set in** | config.yaml | Command line |
| **Changed** | Rarely (requires retraining) | Often (experiment) |
| **Affects** | NMF decomposition | K-means clustering |
| **Question it answers** | "What frequency patterns exist?" | "When do activations change?" |
| **Example value** | 6 | 3 |

---

## Your Scenario: Multiple Submarines

### Config (Fixed)
```yaml
nmf:
  n_components: 6    # Learn 6 freq signatures (sub A, B, C, ambient, etc.)
```

### Command (Flexible)
```bash
python -m src temporal-cluster --input data/ --n-patterns 5
                                            # Find 5 time periods
                                            # (shows when subs overlap)
```

**Result:** 
- 6 frequency signatures describe each submarine's acoustic characteristics
- 5 time periods show exactly when they emit together/apart

Both needed for complete understanding. ✅

---

## Key Insight

**n_components** = "What tools do we have?" (vocabulary)
**n-patterns** = "How are they combined?" (grammar)

You need both vocabulary AND grammar to understand language.
You need both components AND patterns to understand temporal mixture.


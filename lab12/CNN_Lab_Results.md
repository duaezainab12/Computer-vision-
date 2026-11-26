# CNN Feature Visualization, Parameter Audit & Critical Analysis: Results

**Student Name:** Dua e Zainab
**Registration / Roll No.:** FA34-BAI-027
**Notebook:** `CNN_Feature_Visualization_Lab.ipynb`
**Input image:** synthetic building facade, 256×256 grayscale (`building.jpg` was not found, so the fallback image was used)

> All values below are copied from the printed output of the executed notebook.

---

## 1. Part I: Filter Pipeline Results

### 1.1 Kernels used (3×3)

| Filter | Kernel | Sum of weights |
|---|---|---|
| Horizontal Edge (Sobel) | `[[-1,-2,-1],[0,0,0],[1,2,1]]` | 0 |
| Vertical Edge (Sobel) | `[[-1,0,1],[-2,0,2],[-1,0,1]]` | 0 |
| Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | 1 |
| Gaussian Blur | `[[1,2,1],[2,4,2],[1,2,1]] / 16` | 1 |
| Emboss | `[[-2,-1,0],[-1,1,1],[0,1,2]]` | 1 |

Check: the NumPy `conv2d` matches `cv2.filter2D`.

### 1.2 Stage shapes and statistics

| Stage | Operation | Output shape |
|---|---|---|
| A | Raw convolution | 256×256 |
| B | ReLU | 256×256 |
| C | Max Pool 2×2, stride 2 | 128×128 |
| D | Average Pool 2×2, stride 2 | 128×128 |

| Filter | A min | A max | % negative in A | % exact-zero in B |
|---|---|---|---|---|
| Horizontal Edge | −2.922 | 2.914 | 48.535 | 51.648 |
| Vertical Edge | −2.631 | 2.525 | 51.009 | 54.739 |
| Sharpen | −1.290 | 1.929 | 9.464 | 9.464 |
| Gaussian Blur | 0.120 | 0.933 | 0.000 | 0.000 |
| Emboss | −2.467 | 3.098 | 12.076 | 12.112 |

Output figure: `cnn_feature_grid_5x4.png` (5 filters × 4 stages).

---

## 2. Part II: Parameter Audit

Formulas: Conv = (Kh × Kw × Cin + 1) × Cout; Dense = (Nin + 1) × Nout. Pooling, ReLU and GAP have 0 parameters.

Architecture: Input 32×32×3 → Conv1 (5) → Pool → Conv2 (10) → Pool → GAP → Dense(10).

### 2.1 Manual calculation

| Layer | Model A (3×3) | Model B (5×5) |
|---|---|---|
| Conv1 | (3×3×3+1)×5 = **140** | (5×5×3+1)×5 = **380** |
| Conv2 | (3×3×5+1)×10 = **460** | (5×5×5+1)×10 = **1260** |
| Dense | (10+1)×10 = **110** | (10+1)×10 = **110** |
| **Total** | **710** | **1750** |

### 2.2 Audit table

| Model | Kernel | Pooling | Manual Count | Code Output | Match |
|---|---|---|---|---|---|
| Model A (Standard) | 3×3 | Max (2×2, s2) | 710 | 710 | ✔ |
| Model B (Large Kernel) | 5×5 | Avg (2×2, s2) | 1750 | 1750 | ✔ |

### 2.3 Layer shapes (both models)

| Layer | Output shape |
|---|---|
| conv1 | (None, 32, 32, 5) |
| pool1 | (None, 16, 16, 5) |
| conv2 | (None, 16, 16, 10) |
| pool2 | (None, 8, 8, 10) |
| gap | (None, 10) |
| dense_out | (None, 10) |

Keras summary: Model A has 710 trainable parameters (2.77 KB) and Model B has 1,750 (6.84 KB). Non-trainable parameters: 0.

---

## 3. Part III: Evidence for Questions 1–5

### Q1: ReLU blackout and activation alternatives

| Row | Negative in A | Exactly 0 in A | Zero in B | Most negative / positive |
|---|---|---|---|---|
| Horizontal Edge | 48.5% | 3.1% | 51.6% | −2.92 / +2.91 |
| Vertical Edge | 51.0% | 3.7% | 54.7% | −2.63 / +2.53 |

Step-edge direction test:

| Test | Response |
|---|---|
| Horizontal kernel, dark above / bright below | +4 |
| Horizontal kernel, bright above / dark below | −4 |
| Vertical kernel, dark left / bright right | +4 |
| Vertical kernel, bright left / dark right | −4 |

- Sigmoid of a flat (0) response = **0.50** (mid-grey, not black).
- Leaky ReLU (α = 0.1) minimum = **−0.292**; ReLU minimum = 0.

### Q2: Max pooling vs average pooling

| Metric | Horizontal Edge | Vertical Edge | Sharpen |
|---|---|---|---|
| Stage B max | 2.914 | 2.525 | 1.929 |
| Stage C (Max Pool) max | 2.914 | 2.525 | 1.929 |
| Stage D (Avg Pool) max | 2.525 | 1.274 | 1.358 |
| Avg peak as % of Max peak | 86.64 | 50.43 | 70.38 |
| Mean C / Mean D | 0.338 / 0.195 | 0.268 / 0.123 | 0.797 / 0.617 |
| Std (contrast) C / D | 0.757 / 0.475 | 0.538 / 0.253 | 0.411 / 0.334 |
| % pixels above 50% of B-max, C / D | 10.535 / 3.430 | 7.172 / 0.024 | 32.581 / 17.175 |

Dark ink on white background test:

| | Ink amount per pixel | Ink lost |
|---|---|---|
| Original | 1.041% | n/a |
| Max Pool | 0.433% | 58% |
| Avg Pool | 1.041% | 0% |

- Pixels darker than white: Original 1.79% / Max 1.20% / Avg 2.47%.
- 2×2 patch with one ink pixel: max = 255, mean = 191.25.

### Q3: Parameter explosion and receptive field

| Item | Result |
|---|---|
| Model A / Model B | 710 / 1750 |
| Increase | (1750 − 710) / 710 × 100 = **146.48%** (×2.465) |
| 1 / 2 / 3 stacked 3×3 layers | RF 3×3 / **5×5** / 7×7 |
| General saving, equal channels | 25C² vs 18C² → **28%** |
| Example C = 10 | 2510 vs 1820 → saves 690 (27.5%) |

| Layer (this lab) | Single 5×5 | Two 3×3 | Difference |
|---|---|---|---|
| Conv1 (3→5) | 380 | 370 | saves 10 |
| Conv2 (5→10) | 1260 | 1370 | costs 110 more |
| Whole network | 1750 | 1850 | costs 100 more |

### Q4: Order of ReLU and MaxPool

Patch `[[-3, 2], [-1, 4]]`:

| Order | Calculation | Result |
|---|---|---|
| ReLU → MaxPool | max(0, 2, 0, 4) | 4 |
| MaxPool → ReLU | ReLU(max(−3, 2, −1, 4)) = ReLU(4) | 4 |

All-negative patch: both orders give 0.

| Filter | Max-pool orders identical? | Max difference |
|---|---|---|
| Horizontal Edge | True | 0.0 |
| Vertical Edge | True | 0.0 |
| Sharpen | True | 0.0 |
| Gaussian Blur | True | 0.0 |
| Emboss | True | 0.0 |

- Average-pool orders differ: max difference 0.0775 (Horizontal) and 0.6402 (Vertical).
- ReLU evaluations: Conv→ReLU→Pool = **65,536**; Conv→Pool→ReLU = **16,384** (4× fewer).

### Q5: Filter redundancy

| Initialization | Mean filter cosine similarity | Mean feature-map correlation |
|---|---|---|
| Constant (identical) | +1.000 | +1.000 |
| He-normal random | +0.056 | −0.141 |

| Bias | % ReLU outputs that are zero | Outcome |
|---|---|---|
| 0.0 | 0.0% | gradient flows |
| −5.0 | 100.0% | dead filter (gradient 0 everywhere) |

---

## 4. Summary

| Item | Result |
|---|---|
| Model A trainable parameters | 710 (manual = Keras) |
| Model B trainable parameters | 1750 (manual = Keras) |
| Increase A → B | 146.48% |
| 3×3 layers needed for a 5×5 receptive field | 2 |
| Conv→ReLU→MaxPool equals Conv→MaxPool→ReLU | Yes for Max Pool, no for Avg Pool |
| Saved figure | `cnn_feature_grid_5x4.png` |

# Lab 03 — Edge Detection Techniques and Their Impact on Classification Performance

Skin-lesion classification (ISIC, 4 classes) using **raw**, **filtered**, and **edge-map** inputs.
This lab builds on Lab 01 (transfer learning) and Lab 02 (spatial filtering).

**Classes:** Basal cell carcinoma (BCC), Melanoma, Nevus, Pigmented benign keratosis (PBK)
**Image size:** 224 × 224 | **Split:** 1,306 train / 327 validation / 64 test (16 per class)
**Notebook:** `Lab03_Edge_Detection.ipynb` (Google Colab, GPU)

---

## 1. What this lab does

| Task | What was done |
|---|---|
| 1 | Sobel (Gx, Gy, magnitude), Prewitt, Laplacian, LoG and Canny applied to one image per class |
| 2 | Gaussian and salt-and-pepper noise, then edge detection on original / noisy / Gaussian-filtered / median-filtered images (Table 1) |
| 3 | Canny threshold and kernel-size analysis, best configuration selected on validation data (Table 2) |
| 4 | Three datasets built with identical splits: **Set A** raw, **Set B** filtered, **Set C** edge |
| 5 | SVM, Random Forest, KNN, ResNet18 and a small CNN trained on all three sets (Table 3) |
| 6 | Confusion matrices and a metric bar chart for the best model |

---

## 2. Methodology

**Edge detectors** (OpenCV, on grayscale): Sobel 3×3 (Gx, Gy, magnitude); Prewitt 3×3 (magnitude); Laplacian 3×3 (absolute response); LoG (Gaussian σ = 1.5, then Laplacian, absolute response); Canny (Gaussian smoothing, then `cv2.Canny`, default 50/150 with 5×5 smoothing for Tasks 1–2).

**Noise:** Gaussian (σ = 25 on a 0–255 scale) and salt-and-pepper (5% of pixels). **Filters:** Gaussian 5×5 and median 5×5.

**How Table 1 is scored** (20 test images, 5 per class, same noise realisation for every condition):
- Each noisy/filtered edge map is compared with the *same detector on the clean image*. The binarisation threshold is fixed from the clean image (Otsu), so noise cannot hide itself by shifting the threshold. Canny is already binary.
- **F1 vs clean edges:** edge-pixel F1 with a 1-pixel localisation tolerance.
- **False edges (%):** share of detected edge pixels that are not near a clean edge.
- **Segments:** number of connected edge components (more segments for the same edges means more broken edges).
- **Edge Quality** label: F1 ≥ 0.60 Good, ≥ 0.35 Fair, else Poor. **Noise Sensitivity** label: false edges < 25% Low, < 50% Medium, otherwise High. These cut-offs are heuristics. The **Observations** column is auto-generated from the numbers and should be combined with your own reading of the figures.

**Canny selection (Table 2):** each configuration produces edge maps for the train and validation sets; a HOG + RBF-SVM is trained and scored on **validation** accuracy (the test set is not used for selection). Canny-4 uses 50/150 because the lab sheet leaves its thresholds blank.

**Classification setup (identical for all sets):**
- **Set A** raw RGB. **Set B** Gaussian 5×5 (best filter from Lab 02). **Set C** Canny edge map (config selected in Task 3), replicated to 3 channels for the CNNs.
- **Classical models (SVM, RF, KNN):** HOG features (9 orientations, 16×16 cells, 128×128 grayscale → 1,764 features). SVM: RBF, C = 10; RF: 300 trees; KNN: k = 5.
- **CNN Model 1:** ImageNet ResNet18, frozen backbone, new 4-class head. **CNN Model 2:** small 4-block CNN trained from scratch.
- CNNs: 10 epochs, Adam, learning rate 0.001, batch size 32, best-validation epoch kept, seed 42.
- **Inference time** is end-to-end per image (filtering or Canny + HOG where used + model). **Training time** is model fitting only.

---

## 3. Results

### Task 1 — Comparative edge detection

Original → Sobel → Prewitt → Laplacian → LoG → Canny for one image per class:

![Task 1](figures/task1_edge_comparison.png)

Sobel Gx, Gy and gradient magnitude:

![Sobel components](figures/task1_sobel_gx_gy_mag.png)

Sobel and Prewitt give almost identical maps. Laplacian and LoG respond to fine texture and hair. With the default thresholds Canny is very sparse on these images: it picks up the strongest hairs and a few short segments, and returns an almost empty map for the PBK example. This matters for Task 3.

### Task 2 — Effect of noise

Gaussian noise (rows: input, Sobel, Prewitt, Laplacian, LoG, Canny; columns: original, noisy, noisy + Gaussian filter, noisy + median filter):

![Gaussian noise](figures/task2_noise_gaussian.png)

Salt-and-pepper noise:

![Salt and pepper noise](figures/task2_noise_salt_pepper.png)

#### Table 1. Effect of Noise and Preprocessing on Edge Detection (mean over 20 test images)

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations | F1 vs clean edges | False edges (%) | Edge density (%) | Segments |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| Sobel | Original | None | None | Reference (clean) | — | Clean-image reference: 13.6% edge pixels, 307 segments | — | — | 13.62 | 306.9 |
| Sobel | Noisy | Gaussian | None | Fair | High | many false edges; edges much more fragmented/broken than clean | 0.451 | 68.3 | 79.06 | 157.2 |
| Sobel | Noisy | Salt & Pepper | None | Fair | High | many false edges; edges much more fragmented/broken than clean | 0.562 | 57.3 | 40.16 | 437.8 |
| Sobel | Noisy | Gaussian | Gaussian Filter | Good | Medium | some false edges; edges much more fragmented/broken than clean | 0.633 | 46.8 | 36.41 | 413.7 |
| Sobel | Noisy | Salt & Pepper | Median Filter | Good | Low | few false edges; edges less fragmented than clean (smoothing merged/removed detail); weak/fine edges lost (recall 0.60) | 0.715 | 0.0 | 6.66 | 82.0 |
| Prewitt | Original | None | None | Reference (clean) | — | Clean-image reference: 13.6% edge pixels, 311 segments | — | — | 13.63 | 311.2 |
| Laplacian | Original | None | None | Reference (clean) | — | Clean-image reference: 8.7% edge pixels, 542 segments | — | — | 8.68 | 541.7 |
| LoG | Noisy | Gaussian | Gaussian Filter | Fair | High | many false edges; edges much more fragmented/broken than clean | 0.571 | 56.7 | 40.59 | 765.5 |
| Canny | Original | None | Built-in smoothing | Reference (clean) | — | Clean-image reference: 0.7% edge pixels, 6 segments | — | — | 0.71 | 5.9 |
| Canny | Noisy | Gaussian | Gaussian Filter | Poor | Medium | some false edges; edges less fragmented than clean (smoothing merged/removed detail); weak/fine edges lost (recall 0.30) | 0.301 | 29.8 | 0.51 | 6.0 |
| Canny | Noisy | Salt & Pepper | Median Filter | Poor | Low | few false edges; edges less fragmented than clean (smoothing merged/removed detail); weak/fine edges lost (recall 0.20) | 0.230 | 10.0 | 0.33 | 3.0 |

> "Built-in smoothing" for Canny means the Gaussian smoothing stage inside the Canny procedure (OpenCV's `cv2.Canny` does not smooth by itself, so the notebook applies it first).

#### Supplementary Table 1b. F1 vs clean edges for every detector, noise and filter (higher = more robust)

| Detector | Gaussian noise: no filter | Gaussian noise: Gaussian filter | Gaussian noise: median filter | Salt & pepper: no filter | Salt & pepper: Gaussian filter | Salt & pepper: median filter |
|---|---:|---:|---:|---:|---:|---:|
| Sobel | 0.451 | 0.633 | 0.621 | 0.562 | 0.552 | 0.715 |
| Prewitt | 0.450 | 0.636 | 0.626 | 0.561 | 0.553 | 0.722 |
| Laplacian | 0.353 | 0.367 | 0.501 | 0.479 | 0.336 | 0.524 |
| LoG | 0.516 | 0.571 | 0.588 | 0.527 | 0.548 | 0.774 |
| Canny | 0.165 | 0.301 | 0.273 | 0.096 | 0.267 | 0.230 |

### Task 3 — Canny parameter analysis

![Canny configurations](figures/task3_canny_configs.png)

#### Table 2. Canny Parameter Analysis

*Number of Detected Edges = mean edge pixels per 224×224 image (train set). Val accuracy = HOG + SVM on Canny maps, validation set.*

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges | Observation | Edge density (%) | Segments/image | Val accuracy (%) |
|---|---:|---:|---|---|---:|---|---:|---:|---:|
| Canny-1 | 30 | 100 | 3×3 | Balanced | 1925 | 3.8% edge pixels, ~34 segments/image | 3.84 | 33.8 | **43.73** |
| Canny-2 | 50 | 150 | 3×3 | Sparse / under-detected | 729 | 1.5% edge pixels, ~11 segments/image | 1.45 | 10.8 | 40.67 |
| Canny-3 | 100 | 200 | 3×3 | Sparse / under-detected | 271 | 0.5% edge pixels, ~4 segments/image | 0.54 | 4.2 | 36.09 |
| Canny-4 | 50 | 150 | 5×5 | Sparse / under-detected | 412 | 0.8% edge pixels, ~6 segments/image | 0.82 | 5.9 | 40.06 |

**Selected configuration: Canny-1 (low = 30, high = 100, Gaussian 3×3)**, the highest validation accuracy and the only one that is not sparse. The gaps between configurations are small (3 to 8 points on 327 validation images), so this is a modest preference, not a decisive one.

### Tasks 4–5 — Classification on raw, filtered and edge images

#### Table 3. Cross-Lab Classification Performance Comparison (test set, 64 images)

*Accuracy columns come from re-running each representation in this notebook with the same split and epochs (the "Lab 1" and "Lab 2" labels follow the lab sheet). Precision, Recall, F1, Training Time and Inference Time are for the Edge set (Set C). All sets are listed in the next table.*

| Model / Classifier | Accuracy Raw (Lab 1) | Accuracy Filtered (Lab 2) | Accuracy Edge (Lab 3) | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | 43.75 | 45.31 | 42.19 | 43.01 | 42.19 | 40.84 | 2.10 | 4.782 |
| Random Forest | 43.75 | 46.88 | 37.50 | 36.98 | 37.50 | 35.33 | 6.58 | 3.706 |
| KNN | 40.62 | 37.50 | 34.38 | 34.76 | 34.38 | 34.41 | 0.02 | 2.784 |
| CNN Model 1 (ResNet18) | **64.06** | 60.94 | 42.19 | 41.60 | 42.19 | 40.89 | 20.37 | 1.518 |
| CNN Model 2 (Small CNN) | 50.00 | 57.81 | 46.88 | 42.84 | 46.88 | 42.31 | 17.70 | 0.828 |

#### Table 3b. Full results, all models and all sets (test set; metrics are weighted averages)

| Model | Set | Val Acc | Accuracy | Precision | Recall | F1 | Macro-F1 | Train time (s) | Inference (ms/img) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | Raw | 0.4862 | 0.4375 | 0.4554 | 0.4375 | 0.4399 | 0.4399 | 2.08 | 5.061 |
| SVM | Filtered | 0.4557 | 0.4531 | 0.4820 | 0.4531 | 0.4491 | 0.4491 | 2.11 | 4.806 |
| SVM | Edge | 0.4373 | 0.4219 | 0.4301 | 0.4219 | 0.4084 | 0.4084 | 2.10 | 4.782 |
| Random Forest | Raw | 0.4709 | 0.4375 | 0.4453 | 0.4375 | 0.4182 | 0.4182 | 15.98 | 4.261 |
| Random Forest | Filtered | 0.4557 | 0.4688 | 0.4780 | 0.4688 | 0.4357 | 0.4357 | 14.12 | 4.487 |
| Random Forest | Edge | 0.4526 | 0.3750 | 0.3698 | 0.3750 | 0.3533 | 0.3533 | 6.58 | 3.706 |
| KNN | Raw | 0.3945 | 0.4062 | 0.3734 | 0.4062 | 0.3665 | 0.3665 | 0.03 | 3.275 |
| KNN | Filtered | 0.3609 | 0.3750 | 0.3469 | 0.3750 | 0.3410 | 0.3410 | 0.02 | 3.346 |
| KNN | Edge | 0.4098 | 0.3438 | 0.3476 | 0.3438 | 0.3441 | 0.3441 | 0.02 | 2.784 |
| CNN Model 1 (ResNet18) | Raw | 0.6972 | 0.6406 | 0.6667 | 0.6406 | 0.6348 | 0.6348 | 20.78 | 1.056 |
| CNN Model 1 (ResNet18) | Filtered | 0.7064 | 0.6094 | 0.6311 | 0.6094 | 0.5950 | 0.5950 | 19.95 | 1.248 |
| CNN Model 1 (ResNet18) | Edge | 0.4526 | 0.4219 | 0.4160 | 0.4219 | 0.4089 | 0.4089 | 20.37 | 1.518 |
| CNN Model 2 (Small CNN) | Raw | 0.6606 | 0.5000 | 0.5001 | 0.5000 | 0.4929 | 0.4929 | 17.38 | 0.425 |
| CNN Model 2 (Small CNN) | Filtered | 0.6575 | 0.5781 | 0.5857 | 0.5781 | 0.5432 | 0.5432 | 17.72 | 0.584 |
| CNN Model 2 (Small CNN) | Edge | 0.3700 | 0.4688 | 0.4284 | 0.4688 | 0.4231 | 0.4231 | 17.70 | 0.828 |

**Accuracy change relative to raw (percentage points):**

| Model | Filtered − Raw | Edge − Raw |
|---|---:|---:|
| SVM | +1.56 | −1.56 |
| Random Forest | +3.12 | −6.25 |
| KNN | −3.12 | −6.25 |
| CNN Model 1 (ResNet18) | −3.12 | −21.88 |
| CNN Model 2 (Small CNN) | +7.81 | −3.12 |
| **Mean accuracy per representation** | Filtered 49.69 | Edge 40.62 (Raw 48.44) |

### Task 6 — Best model: CNN Model 1 (ResNet18)

Best mean accuracy across the three representations (55.73%).

![Confusion matrices](figures/task6_confusion_matrices.png)

![Metric comparison](figures/task6_metric_bars.png)

| Input | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Raw | 64.06 | 66.67 | 64.06 | 63.48 |
| Filtered | 60.94 | 63.11 | 60.94 | 59.50 |
| Edge | 42.19 | 41.60 | 42.19 | 40.89 |

---

## 4. Discussion

**Q1 — Which detector was most sensitive to noise?**
The answer depends on what you measure, so check it against the figures.
- *Spurious edges:* Sobel's edge density rises from 13.6% to 79.1% under Gaussian noise and to 40.2% under salt-and-pepper noise, with 57–68% of detected pixels being false. Sobel and Prewitt behave alike, and Laplacian has the lowest F1 of the derivative operators (0.353 for Gaussian noise, no filter).
- *F1 vs clean edges:* Canny scores lowest (0.130 averaged over both noise types), but this comes from **missed** edges (recall 0.20–0.30), not false ones (10–30%). Canny's edge map here is very sparse (0.7% density) because the fixed 50/150 thresholds are high for these images. Extra smoothing lowers gradient magnitudes below the thresholds, so weak edges vanish. Read this as a threshold-tuning effect, not as Canny being intrinsically the noisiest.

**Q2 — Effect of Gaussian and median filtering.**
Both improved most detectors. Under Gaussian noise the two filters helped Sobel and Prewitt about equally (F1 gain ≈ +0.17 to +0.19), while median filtering helped Laplacian much more (+0.148 vs +0.014). Under salt-and-pepper noise **median filtering was clearly better** for Sobel (+0.153 vs −0.011), Prewitt, Laplacian (+0.045 vs −0.143) and LoG (+0.247 vs +0.021); Gaussian filtering barely helped or even hurt, because it spreads the extreme pixels instead of removing them. Canny was the exception: Gaussian filtering helped slightly more (+0.171 vs +0.134).

**Q3 — Canny thresholds.**
Raising the thresholds cut the number of edges sharply: 1,925 → 729 → 271 edge pixels per image for 30/100 → 50/150 → 100/200 (3×3). Low thresholds keep more detail but also more texture and hair. High thresholds leave sparse, mostly disconnected fragments. A 5×5 kernel at 50/150 gave 412 edges versus 729 for 3×3, since stronger smoothing suppresses weak edges. Validation accuracy rose as more edges were kept (43.7% at 30/100 vs 36.1% at 100/200).

**Q4 — Edge-only vs raw accuracy.**
Edge images **reduced accuracy for all five models**: −1.6 (SVM), −6.3 (RF), −6.3 (KNN), −21.9 (ResNet18) and −3.1 points (small CNN). Mean accuracy fell from 48.4% (raw) to 40.6% (edge). Likely reasons: colour and internal texture are removed; hair and other artifacts create strong edges that are unrelated to diagnosis; a sparse binary edge map has little intensity information; and the pretrained ResNet18 was trained on natural colour images, so edge maps are very unlike its training data (its −21.9 point drop is the largest). The small CNN, trained from scratch, lost the least among the CNNs.

**Q5 — Information lost.**
Colour (variegation and hue are diagnostic in melanoma), fine texture such as pigment patterns, intensity and shading that indicate depth and pigment concentration, and gradual transitions that a thin edge does not capture. What remains is mainly border shape and contour.

**Q6 — Learned vs manual features.**
A CNN learns edge-like filters in its early layers but also colour, texture and higher-level patterns, and it combines them for the task. It needs no hand-picked thresholds or kernel sizes, and can adapt to the data (for example, ignoring hair). Here the raw-input ResNet18 (64.1%) beat every classical model on any input (best 46.9%).

**Q7 — Best representation.**
Raw and filtered images were roughly tied, and both were clearly better than edge images. Filtered had the higher mean accuracy (49.7% vs 48.4%), and filtering helped three of five models, but the mean difference is about one test image, so it is not reliable evidence. The best single result was ResNet18 on **raw** images (64.1%; its validation accuracy on filtered images was equally high). Edge images were the weakest representation.

---

## 5. Limitations

- **Small test set:** 64 images, so one image equals 1.56 percentage points. Differences of a few points between raw and filtered are within noise. Only one random seed was run.
- **Classical models use grayscale HOG features**, so they never see colour in any set, while the CNNs receive RGB for Sets A and B.
- **Table 1 labels are heuristic.** The Edge Quality and Noise Sensitivity words come from the thresholds in Section 2. Pair them with a visual reading of the Task 2 figures.
- **Task 2 uses fixed Canny thresholds (50/150, 5×5)**, which turned out to be sparse on this dataset (Table 2). Task 4 uses the better validated 30/100 (3×3) setting.
- **Not identical to Labs 01/02:** the raw/filtered numbers are re-run here with a different pipeline (data split, feature setup, ImageNet normalisation), so they will not match those labs' README numbers.
- Results are experimental and not clinical.

---

## 6. Reproduce

1. Open `Lab03_Edge_Detection.ipynb` in Google Colab and set **Runtime → GPU**.
2. Add your Kaggle token as a Colab **Secret** named `KAGGLE_API_TOKEN` (never paste it into a cell).
3. Run all cells in order. Tables are saved as CSV, figures as PNG, and everything is zipped in `/content/lab03_outputs.zip`.

Dataset: [Skin Cancer ISIC (Kaggle)](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

## 7. Files

```
Lab03/
├── README.md
├── Lab03_Edge_Detection.ipynb
└── figures/
    ├── task1_edge_comparison.png
    ├── task1_sobel_gx_gy_mag.png
    ├── task2_noise_gaussian.png
    ├── task2_noise_salt_pepper.png
    ├── task3_canny_configs.png
    ├── task6_confusion_matrices.png
    └── task6_metric_bars.png
```

# Computer Vision Lab Mid: HOG vs SIFT on Blood-Cell Images

**Name:** Dua e Zainab  |  **Reg No:** FA34-BAI-027  |  **Course:** Computer Vision Lab  |  **Marks:** 25

Two classical pipelines, **HOG + LinearSVC** and **SIFT (fixed grid) + LinearSVC**, are compared on four BloodMNIST cell types. Both are tested on clean, rotated (20 degrees) and noisy (sigma = 0.05) images. All numbers below are copied from the executed notebook `FA34_BAI_027_DuaeZainab_LabMid.ipynb`.

---

## 1. Headline results

| Model | Features | Best C | Val accuracy | Val macro-F1 | Test clean | Test rotated 20° | Test noisy σ=0.05 |
|---|---|---|---|---|---|---|---|
| HOG + SVM | 1764 | 0.1 | 0.7250 | 0.7243 | 0.6333 | 0.6750 | **0.2500** |
| SIFT grid + SVM | 4608 | 0.1 | **0.7500** | **0.7479** | **0.7000** | 0.6750 | **0.7333** |

---

## 2. Personal settings (registration FA34-BAI-027)

| Setting | Rule | My value |
|---|---|---|
| Random seed | last two digits of reg no. | **27** |
| HOG cell size | last digit 7 is odd | **16 x 16** |
| SIFT grid size | last digit 7 is in 5 to 9 | **6 x 6 (36 keypoints)** |

### Fixed settings (same for everyone)

| Item | Value |
|---|---|
| Dataset | BloodMNIST (`medmnist`), size 64 |
| Classes | eosinophil (1), monocyte (5), neutrophil (6), platelet (7) |
| Images per class | 100 train, 30 validation, 30 test (chosen with seed 27) |
| Image preparation | Grayscale, resize to 128 x 128 |
| HOG | 9 orientations, 2 x 2 cells per block, L2-Hys |
| Classifier | StandardScaler + LinearSVC, C chosen from {0.1, 1, 10} on validation |
| Rotation test | 20 degrees (border reflect) |
| Noise test | Gaussian, sigma = 0.05 (image scaled to 0 to 1) |

---

## 3. Q1: Data setup

| Split | X shape | y shape | eosinophil | monocyte | neutrophil | platelet |
|---|---|---|---|---|---|---|
| Train | (400, 128, 128) | (400,) | 100 | 100 | 100 | 100 |
| Validation | (120, 128, 128) | (120,) | 30 | 30 | 30 | 30 |
| Test | (120, 128, 128) | (120,) | 30 | 30 | 30 | 30 |

![Q1 examples](images/q1_examples.png)

---

## 4. Q2: HOG + SVM

### Feature length (by hand, matches the printed 1764)

| Step | Calculation | Result |
|---|---|---|
| Cells per side | 128 / 16 | 8 |
| Blocks per side (2 x 2 cells, stride 1) | 8 − 2 + 1 | 7 |
| Blocks in total | 7 x 7 | 49 |
| Values per block | 2 x 2 x 9 orientations | 36 |
| **Feature length** | 49 x 36 | **1764** |

### Choosing C on the validation set

| C | Val accuracy | Val macro-F1 | Selected |
|---|---|---|---|
| **0.1** | 0.7250 | 0.7243 | yes |
| 1 | 0.7250 | 0.7239 | |
| 10 | 0.7250 | 0.7239 | |

### Validation confusion matrix (rows = true, columns = predicted)

| True \ Predicted | eosinophil | monocyte | neutrophil | platelet | Correct | Recall |
|---|---|---|---|---|---|---|
| eosinophil | **15** | 10 | 5 | 0 | 15 / 30 | 50.0% |
| monocyte | 5 | **22** | 3 | 0 | 22 / 30 | 73.3% |
| neutrophil | 6 | 3 | **21** | 0 | 21 / 30 | 70.0% |
| platelet | 0 | 1 | 0 | **29** | 29 / 30 | 96.7% |
| **Total** | | | | | **87 / 120** | **72.5%** |

![HOG pictures](images/q2_hog_pictures.png)

![HOG validation confusion matrix](images/q2_hog_confusion_val.png)

---

## 5. Q3: SIFT + SVM

### Normal SIFT (`detectAndCompute`) on the 400 training images

| Statistic | Value |
|---|---|
| Average keypoints per image | 110.4 |
| Minimum | 19 |
| Maximum | 358 |
| Images with zero keypoints | 0 |

**Why this is a problem for an SVM:** each keypoint gives a 128-value descriptor, so an image with more keypoints gives a longer feature vector, and an SVM needs every sample to have the same fixed length.

![SIFT keypoints](images/q3_sift_keypoints.png)

### Fixed grid SIFT (feature length by hand, matches the printed 4608)

| Item | Value |
|---|---|
| Grid | 6 x 6 = 36 grid points over 128 x 128 |
| Grid centres (pixels) | 10.7, 32, 53.3, 74.7, 96, 117.3 |
| Keypoint size / angle | 32 / 0 |
| Descriptor per point | 128 |
| **Feature length** | 36 x 128 = **4608** |

### Choosing C on the validation set

| C | Val accuracy | Val macro-F1 | Selected |
|---|---|---|---|
| **0.1** | 0.7500 | 0.7479 | yes |
| 1 | 0.7500 | 0.7479 | |
| 10 | 0.7500 | 0.7479 | |

### Validation confusion matrix (rows = true, columns = predicted)

| True \ Predicted | eosinophil | monocyte | neutrophil | platelet | Correct | Recall |
|---|---|---|---|---|---|---|
| eosinophil | **15** | 11 | 3 | 1 | 15 / 30 | 50.0% |
| monocyte | 7 | **21** | 2 | 0 | 21 / 30 | 70.0% |
| neutrophil | 3 | 2 | **25** | 0 | 25 / 30 | 83.3% |
| platelet | 0 | 1 | 0 | **29** | 29 / 30 | 96.7% |
| **Total** | | | | | **90 / 120** | **75.0%** |

![SIFT validation confusion matrix](images/q3_sift_confusion_val.png)

---

## 6. Q4: Robustness (test set, frozen models)

### Prediction written before running

| Case | Predicted model to lose more | Reason |
|---|---|---|
| Rotation | HOG | Features are tied to fixed gradient directions and cell positions |
| Noise | HOG | Raw pixel gradients with no smoothing; SIFT blurs first |

### Accuracy table

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy | Rotation drop | Noise drop |
|---|---|---|---|---|---|
| HOG + SVM | 0.6333 | 0.6750 | 0.2500 | −0.0417 | 0.3833 |
| SIFT + SVM | 0.7000 | 0.6750 | 0.7333 | 0.0250 | −0.0333 |

### Same results as images correct (out of 120)

| Model | Clean | Rotated | Noisy |
|---|---|---|---|
| HOG + SVM | 76 | 81 | 30 |
| SIFT + SVM | 84 | 81 | 88 |

![clean, rotated, noisy example](images/q4_clean_rotated_noisy.png)

### Test confusion matrices on clean images (rows = true, columns = predicted)

**HOG + SVM**

| True \ Predicted | eosinophil | monocyte | neutrophil | platelet | Recall |
|---|---|---|---|---|---|
| eosinophil | **12** | 7 | 11 | 0 | 40.0% |
| monocyte | 9 | **17** | 4 | 0 | 56.7% |
| neutrophil | 7 | 6 | **17** | 0 | 56.7% |
| platelet | 0 | 0 | 0 | **30** | 100% |

**SIFT + SVM**

| True \ Predicted | eosinophil | monocyte | neutrophil | platelet | Recall |
|---|---|---|---|---|---|
| eosinophil | **17** | 7 | 6 | 0 | 56.7% |
| monocyte | 6 | **20** | 3 | 1 | 66.7% |
| neutrophil | 9 | 2 | **17** | 2 | 56.7% |
| platelet | 0 | 0 | 0 | **30** | 100% |

### Was my prediction correct?

| Case | Prediction | Outcome |
|---|---|---|
| Noise | HOG loses more | **Correct.** HOG fell from 0.6333 to 0.2500 (chance level for 4 classes), SIFT did not drop (0.7000 to 0.7333). |
| Rotation | HOG loses more | **Wrong.** HOG went 0.6333 to 0.6750 and SIFT 0.7000 to 0.6750, but 3 to 5 images out of 120 is within sampling noise, so neither model can be called more rotation-sensitive. |

**Overall: partly correct.**

---

## 7. Q5: Short answers and AI log

### Most confused class pairs (total errors between the two classes, from the confusion matrices)

| Pair | HOG (val) | SIFT (val) | HOG (test) | SIFT (test) |
|---|---|---|---|---|
| eosinophil and monocyte | **15** | **18** | 16 | 13 |
| eosinophil and neutrophil | 11 | 6 | **18** | **15** |
| monocyte and neutrophil | 6 | 4 | 10 | 5 |
| platelet and any other class | 1 | 2 | 0 | 3 |

Eosinophil is the hardest class. Validation recall is only 50.0% for both models, and platelet is almost never confused (29 of 30 on validation, 30 of 30 on test). Grayscale removes the red-orange stain of eosinophil granules, leaving only nucleus shape and texture, which look similar to monocytes and neutrophils.

### Short answers

| Question | Answer |
|---|---|
| (a) Why does HOG use block normalisation? | Gradient magnitudes change with lighting and contrast. Normalising each block makes the descriptor depend on local gradient shape, not brightness, so it is more robust to illumination. |
| (b) Why is it wrong to choose C using the test set? | The test set must act as unseen data. Picking C by test accuracy lets the test set influence the model (data leakage), so the reported score is optimistic. C is chosen on validation only. |

### AI log

| Prompt (one line) | What I changed | How I checked |
|---|---|---|
| "Code for HOG vs SIFT + LinearSVC on BloodMNIST, seed 27, HOG cell 16, 6x6 SIFT grid" | Set my reg-no settings, added StandardScaler and C selection on validation | Printed shapes, feature lengths 1764 and 4608 (matched hand calculation), plotted HOG and keypoint pictures |
| "Check my results and tell me what to fix" | Filled in the Q4 explanation and Q5 confused-class answer using my printed numbers | Re-computed accuracies from the confusion matrices (87/120, 90/120, 76/120, 84/120) |

---

## 8. Repository contents

| File | Description |
|---|---|
| `FA34_BAI_027_DuaeZainab_LabMid.ipynb` | Executed notebook with all outputs |
| `README.md` | This summary |
| `Q1_data_setup.md` | Q1 results |
| `Q2_hog_svm.md` | Q2 results |
| `Q3_sift_svm.md` | Q3 results |
| `Q4_robustness.md` | Q4 results |
| `Q5_short_answers_and_ai_log.md` | Q5 answers and AI log |
| `images/` | Figures used in the result files |

## 9. How to reproduce

| Step | Command / action |
|---|---|
| 1. Install | `pip install medmnist opencv-python scikit-image scikit-learn matplotlib pandas` |
| 2. Run | Open the notebook in Google Colab and choose Runtime → Restart and run all |
| 3. Seed | `SEED = 27`; the same seed gives the same image subsets and noise |

The test set is used only in Q4. Per-class recall and the pair-error counts are computed from the printed confusion matrices.

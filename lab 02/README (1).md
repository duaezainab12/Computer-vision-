# Lab 02 – Effect of Image Filtering on Skin-Lesion Classification

Computer Vision course · Lab Activity 02

This lab investigates how classic spatial-domain image filters affect the performance of **pretrained deep-learning models** on a 4-class skin-lesion classification task (ISIC images). Every experiment uses the same data split, preprocessing, training settings and metrics so the comparison is fair.

**Notebook:** `CV_lab02.ipynb` (built for Google Colab with a GPU runtime)

---

## Table of Contents

1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Experimental Setup](#experimental-setup)
4. [Filters Compared](#filters-compared)
5. [Models](#models)
6. [Results](#results)
7. [Key Findings](#key-findings)
8. [Limitations](#limitations)
9. [How to Run](#how-to-run)
10. [Repository Structure](#repository-structure)
11. [Tech Stack](#tech-stack)

---

## Objective

Measure how **original, mean, Gaussian, median, sharpening and Sobel** filtered inputs change the accuracy, macro-F1, balanced accuracy and AUC of three pretrained CNNs used as frozen feature extractors.

## Dataset

- **Source:** [Skin Cancer ISIC (9 classes) on Kaggle](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) (`nodoubttome/skin-cancer9-classesisic`)
- **Classes used (4 of 9):** melanoma, basal cell carcinoma, nevus, pigmented benign keratosis
- **Image mode:** all RGB; original sizes vary (most common: 600×450, 1024×768), resized to 224×224

| Class | Label | Train | Validation | Test |
|---|:---:|:---:|:---:|:---:|
| basal cell carcinoma | 0 | 301 | 75 | 16 |
| melanoma | 1 | 350 | 88 | 16 |
| nevus | 2 | 286 | 71 | 16 |
| pigmented benign keratosis | 3 | 369 | 93 | 16 |
| **Total** | | **1,306** | **327** | **64** |

The original `Train` folder was split 80/20 into train/validation using a stratified split (`random_state=42`). The original `Test` folder is used as-is for final evaluation.

## Experimental Setup

| Parameter | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Pixel scaling | 0 – 1 |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer / Learning rate | Adam / 0.001 |
| Loss | Sparse categorical cross-entropy |
| Random seed | 42 |
| Pretrained layers | Frozen (only the classification head is trained) |
| Framework | TensorFlow/Keras (VGG16, DenseNet121), PyTorch (ResNet18) |

## Filters Compared

| Filter | Implementation (OpenCV) | Purpose |
|---|---|---|
| Original | No processing | Baseline |
| Mean | `cv2.blur`, 5×5 kernel | Smoothing / noise reduction |
| Gaussian | `cv2.GaussianBlur`, 5×5 kernel | Weighted smoothing |
| Median | `cv2.medianBlur`, size 5 | Noise removal, edge-preserving |
| Sharpen | `cv2.filter2D`, 3×3 kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | Enhance edges and detail |
| Sobel | Gradient magnitude of grayscale (ksize 3), replicated to 3 channels | Edge map only |

## Models

| Model | Framework | Pretrained on | Head | Total params | Trainable params |
|---|---|---|---|---|---|
| VGG16 | TensorFlow/Keras | ImageNet | GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(4, softmax) | 14,780,868 | 66,180 |
| DenseNet121 | TensorFlow/Keras | ImageNet | GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(4, softmax) | 7,169,220 | 131,716 |
| ResNet18 | PyTorch | ImageNet | Linear(512 → 4) | – | Final layer only |

## Results

All values are percentages, evaluated on the 64-image test set.

### Full results (accuracy, macro-F1, AUC)

| Model | Filter | Accuracy | Macro-F1 | AUC |
|---|---|:---:|:---:|:---:|
| VGG16 | original | 48.44 | 49.02 | 73.89 |
| VGG16 | mean | 59.38 | 58.56 | 75.98 |
| VGG16 | gaussian | 56.25 | 55.19 | 76.60 |
| VGG16 | median | 51.56 | 48.98 | 71.00 |
| VGG16 | sharpen | 43.75 | 44.02 | 72.17 |
| VGG16 | sobel | 26.56 | 26.41 | 54.46 |
| DenseNet121 | original | 57.81 | 56.55 | 86.72 |
| DenseNet121 | mean | **64.06** | 62.58 | 85.35 |
| DenseNet121 | gaussian | **64.06** | 61.74 | 85.32 |
| DenseNet121 | median | 60.94 | 60.04 | 82.62 |
| DenseNet121 | sharpen | 54.69 | 51.58 | 82.42 |
| DenseNet121 | sobel | 42.19 | 41.49 | 68.00 |
| ResNet18 | original | 56.25 | 54.41 | 82.23 |
| ResNet18 | mean | 59.38 | 57.57 | 85.58 |
| ResNet18 | gaussian | 60.94 | 58.76 | 86.43 |
| ResNet18 | median | 57.81 | 53.98 | 80.53 |
| ResNet18 | sharpen | **68.75** | **65.21** | 86.52 |
| ResNet18 | sobel | 50.00 | 48.61 | 77.44 |

### Best filter per model

| Model | Best filter | Accuracy | Macro-F1 | Balanced Accuracy | AUC |
|---|---|:---:|:---:|:---:|:---:|
| VGG16 | mean | 59.38 | 58.56 | 59.38 | 75.98 |
| DenseNet121 | mean | 64.06 | 62.58 | 64.06 | 85.35 |
| ResNet18 | sharpen | 68.75 | 65.21 | 68.75 | 86.52 |

### Average performance per filter (across the 3 models)

| Filter | Accuracy | Macro-F1 | Balanced Accuracy | AUC |
|---|:---:|:---:|:---:|:---:|
| mean | 60.94 | 59.57 | 60.94 | 82.30 |
| gaussian | 60.42 | 58.56 | 60.42 | 82.78 |
| median | 56.77 | 54.33 | 56.77 | 78.05 |
| sharpen | 55.73 | 53.60 | 55.73 | 80.37 |
| original | 54.17 | 53.33 | 54.17 | 80.95 |
| sobel | 39.58 | 38.84 | 39.58 | 66.63 |

### Change relative to the original images (percentage points)

| Model | Filter | Δ Accuracy | Δ Macro-F1 | Δ AUC |
|---|---|:---:|:---:|:---:|
| VGG16 | mean | +10.94 | +9.54 | +2.09 |
| VGG16 | gaussian | +7.81 | +6.17 | +2.71 |
| VGG16 | median | +3.12 | -0.04 | -2.89 |
| VGG16 | sharpen | -4.69 | -5.00 | -1.72 |
| VGG16 | sobel | -21.88 | -22.61 | -19.43 |
| DenseNet121 | mean | +6.25 | +6.03 | -1.37 |
| DenseNet121 | gaussian | +6.25 | +5.19 | -1.40 |
| DenseNet121 | median | +3.13 | +3.49 | -4.10 |
| DenseNet121 | sharpen | -3.12 | -4.97 | -4.30 |
| DenseNet121 | sobel | -15.62 | -15.06 | -18.72 |
| ResNet18 | mean | +3.13 | +3.16 | +3.35 |
| ResNet18 | gaussian | +4.69 | +4.35 | +4.20 |
| ResNet18 | median | +1.56 | -0.43 | -1.70 |
| ResNet18 | sharpen | +12.50 | +10.80 | +4.29 |
| ResNet18 | sobel | -6.25 | -5.80 | -4.79 |

## Key Findings

| # | Finding |
|---|---|
| 1 | **Smoothing filters (mean, Gaussian) helped most models**, giving the best average accuracy (~60–61%) versus 54.17% for original images. |
| 2 | **Sobel hurt every model.** Reducing the image to an edge map removes colour and texture that lesion classification depends on (average accuracy 39.58%). |
| 3 | **Sharpening was model-dependent:** best result overall for ResNet18 (+12.5 pts accuracy) but negative for VGG16 and DenseNet121. |
| 4 | **Median filtering** gave small, inconsistent changes. |
| 5 | **DenseNet121 had the strongest AUC** (≈86.7% on original images), while **ResNet18 + sharpen** had the best accuracy and macro-F1. |

## Limitations

| Limitation | Impact |
|---|---|
| Test set has only 64 images (16 per class) | One image is about 1.56 points, so small differences (e.g. ±3 pts) may just be noise. |
| Single run with one seed (42) | No variance estimate across runs. |
| Only 10 epochs with frozen backbones | Models are under-trained; absolute scores are modest. |
| Only 4 of 9 ISIC classes used | Results may not transfer to the full dataset. |
| ResNet18 uses a different framework (PyTorch) | Minor pipeline differences from the Keras models. |

## How to Run

1. Open `CV_lab02.ipynb` in Google Colab and enable a **GPU** runtime.
2. Add your own Kaggle API token as a Colab secret (do **not** paste it into the notebook):
   ```python
   from google.colab import userdata
   import os
   os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
   ```
3. Run all cells in order. The notebook will:
   - download and unzip the dataset,
   - build the train/validation/test splits,
   - apply each filter and train/evaluate VGG16, DenseNet121 and ResNet18,
   - print the final comparison tables.

## Repository Structure

```
.
├── CV_lab02.ipynb   # Full experiment notebook
└── README.md        # Project documentation
```

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Deep learning | TensorFlow / Keras 2.20, PyTorch + Torchvision |
| Image processing | OpenCV, Pillow |
| Data & metrics | NumPy, pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab (GPU) |

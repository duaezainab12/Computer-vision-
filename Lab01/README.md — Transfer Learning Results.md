# Skin Lesion Classification — Transfer Learning

Comparison of pretrained deep learning models for **4-class skin lesion classification**.

**Classes:** Melanoma, Basal Cell Carcinoma, Nevus, Pigmented Benign Keratosis  
**Image Size:** 224 × 224  
**Training:** 10 epochs, Adam optimizer, learning rate 0.001  
**Test Set:** 64 images (16 per class)

## Transfer Learning Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---:|---:|---:|---:|---:|
| AlexNet | 25.00% | 6.25% | 25.00% | 10.00% | 50.00% |
| VGG16 | 37.50% | 46.47% | 37.50% | 37.12% | 75.55% |
| VGG19 | 35.94% | 27.93% | 35.94% | 30.84% | 78.58% |
| **ResNet18** | **67.19%** | **71.06%** | **67.19%** | **64.02%** | **87.70%** |
| ResNet50 | 25.00% | 6.25% | 25.00% | 10.00% | 50.24% |
| ResNet101 | 25.00% | 6.25% | 25.00% | 10.00% | 49.09% |
| DenseNet121 | 53.12% | 54.27% | 53.12% | 49.76% | 80.34% |
| EfficientNet-B0 | 25.00% | 6.25% | 25.00% | 10.00% | 54.33% |

### Best Model

**ResNet18** achieved the best performance among the completed experiments:

- Accuracy: **67.19%**
- F1-Score: **64.02%**
- AUC: **87.70%**

> Note: The test set contains only 64 images, so the results should be considered experimental rather than clinically conclusive.

## 2. Deep Feature Extraction and Classifier Comparison

After comparing different transfer learning architectures, **ResNet18** was selected as the feature extractor for the next experiment. Instead of directly using the neural network for classification, the final classification layer was removed and the learned image representations were extracted as deep features.

These features were then provided to different traditional machine learning classifiers to determine whether they could effectively classify the four selected skin lesion categories.

The following classifiers were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* Linear SVM
* RBF-SVM
* XGBoost

### Results

| Classifier          |   Accuracy |  Precision |     Recall |   F1-Score |        AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     57.81% |     59.04% |     57.81% |     57.94% |     80.11% |
| Decision Tree       |     46.88% |     49.65% |     46.88% |     46.17% |     64.58% |
| Random Forest       |     57.81% |     62.94% |     57.81% |     58.35% |     85.07% |
| KNN                 |     53.12% |     53.43% |     53.12% |     52.51% |     78.97% |
| Linear SVM          |     60.94% |     62.30% |     60.94% |     60.45% |     82.78% |
| RBF-SVM             |     62.50% |     63.98% |     62.50% |     61.33% | **88.74%** |
| **XGBoost**         | **65.62%** | **67.49%** | **65.62%** | **63.93%** |     83.53% |

### Key Findings

**XGBoost achieved the highest classification accuracy (65.62%) and F1-score (63.93%)** when working with ResNet18 deep features. This indicates that the extracted features contained useful information for separating the four lesion categories.

The **RBF-SVM achieved the highest AUC (88.74%)**, showing strong class-separation capability despite having slightly lower accuracy than XGBoost.

Overall, the experiment demonstrates that pretrained CNN features can also be combined effectively with traditional machine learning classifiers.

---

## 3. Computational Efficiency Comparison

In addition to classification performance, the transfer learning models were compared from a computational perspective. This is important when considering the practical deployment of a skin lesion classification system, particularly on systems with limited computational resources.

The comparison considered:

* Number of trainable parameters
* Approximate model size
* Inference time per image
* Classification accuracy

### Results

| Model           | Parameters (M) | Model Size (MB) | Inference Time (ms) |   Accuracy |
| --------------- | -------------: | --------------: | ------------------: | ---------: |
| AlexNet         |         57.020 |         217.515 |           **2.102** |     25.00% |
| VGG16           |        134.277 |         512.226 |               9.809 |     37.50% |
| VGG19           |        139.587 |         532.481 |              10.839 |     35.94% |
| **ResNet18**    |     **11.179** |      **42.643** |               2.927 | **67.19%** |
| ResNet50        |         23.516 |          89.707 |               7.547 |     25.00% |
| ResNet101       |         42.508 |         162.157 |              16.913 |     25.00% |
| DenseNet121     |          6.958 |          26.542 |              24.806 |     53.12% |
| EfficientNet-B0 |      **4.013** |      **15.307** |               9.245 |     25.00% |

### Key Findings

The results show a clear difference between model complexity and classification performance. **VGG16 and VGG19 have the largest number of parameters and model sizes**, while EfficientNet-B0 has the smallest parameter count.

Among the tested architectures, **ResNet18 provided the best overall balance between accuracy and computational requirements**. It achieved the highest accuracy of **67.19%** while using only **11.179 million parameters** and approximately **42.643 MB** of model storage.

This makes ResNet18 a practical candidate for further development and experimentation in this project.

> **Note:** The test set contains only 64 images (16 per class), so these results are experimental and should not be interpreted as clinical performance.

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
| VGG16 | Pending | Pending | Pending | Pending | Pending |
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
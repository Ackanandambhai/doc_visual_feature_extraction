# Document Visual Feature Extraction

A deep learning framework for document image classification using multiple CNN backbones on the Tobacco3482 dataset. The repository provides automated training, evaluation, k-fold cross-validation, performance comparison, confusion matrices, and report generation across several state-of-the-art architectures.

## Supported Backbones

- MobileNetV3-Small
- MobileNetV2
- ResNet18
- ResNet50
- EfficientNet-B0
- DenseNet121
- VGG16

---

## Project Structure

```text
doc_visual_feature_extraction/
│
├── data/
│   └── raw/
│       └── Tobacco3482/
│
├── scripts/
│   └── run_experiment.sh
│
├── src/
│   ├── main.py
├── outputs/
│   ├── mobilenet_v3_small/
│   ├── mobilenet_v2/
│   ├── resnet18/
│   ├── resnet50/
│   ├── efficientnet_b0/
│   ├── densenet121/
│   └── vgg16/
│
└── README.md
```

---

## Run All Backbones

```bash
bash scripts/run_experiment.sh
```

---

## Run a Specific Backbone

### VGG16

```bash
caffeinate -dimsu bash scripts/run_experiment.sh \
    --backbones vgg16
```

### ResNet50

```bash
caffeinate -dimsu bash scripts/run_experiment.sh \
    --backbones resnet50
```

### DenseNet121

```bash
caffeinate -dimsu bash scripts/run_experiment.sh \
    --backbones densenet121
```

> macOS users can use `caffeinate -dimsu` to prevent sleep during long training runs.

---

## Dataset

Place the Tobacco3482 dataset in:

```text
data/raw/Tobacco3482/
```

---

## Generated Outputs

- Cross-validation results
- Classification reports
- Confusion matrices
- Training and validation curves
- Accuracy, Precision, Recall and F1-score metrics
- Model comparison plots
- Performance summaries

---

## Training Pipeline

```text
Input Document Images
          │
          ▼
Data Loading & Preprocessing
          │
          ▼
K-Fold Cross Validation
          │
          ▼
Backbone Selection
          │
          ▼
Model Training
          │
          ▼
Validation & Testing
          │
          ▼
Performance Evaluation
          │
          ▼
Reports, Plots & Confusion Matrices
```

## Model Performance Summary

| Model | Parameters | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---------|-----------:|---------------:|--------------:|----------:|-------:|---------:|
| densenet121 | 6,964,106 | 0.9673 | 0.8561 | 0.8552 | 0.8296 | 0.8387 |
| efficientnet_b0 | 4,020,358 | 0.8916 | 0.8372 | 0.8294 | 0.8218 | 0.8242 |
| mobilenet_v2 | 2,236,682 | 0.9370 | 0.8409 | 0.8405 | 0.8175 | 0.8255 |
| mobilenet_v3_small | 1,528,106 | 0.8356 | 0.7685 | 0.7826 | 0.7415 | 0.7543 |
| resnet18 | 11,181,642 | 0.9364 | 0.8271 | 0.8283 | 0.7991 | 0.8091 |
| resnet50 | 23,528,522 | 0.9749 | 0.8667 | 0.8590 | 0.8460 | 0.8513 |
| vgg16 | 134,301,514 | 0.9321 | 0.8277 | 0.8209 | 0.7935 | 0.8036 |

# Document Visual Feature Extraction

A deep learning framework for **feature extraction** and  **document image classification** using multiple CNN backbones on the **Tobacco3482** dataset. The project supports automated training, k-fold cross-validation, evaluation, model comparison, confusion matrices, and visualization of results across several state-of-the-art architectures.

---

## Project Architecture

<img width="1600" height="956" alt="dvfe_architecture" src="https://github.com/user-attachments/assets/4b238950-6e44-45c7-a87d-273b78739664" />

# Supported Backbones

* MobileNetV3-Small
* MobileNetV2
* ResNet18
* ResNet50
* EfficientNet-B0
* DenseNet121
* VGG16

---

# Project Structure

```text
doc_visual_feature_extraction/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│       └── Tobacco3482/
│
├── outputs/
│   ├── mobilenet_v3_small/
│   ├── mobilenet_v2/
│   ├── resnet18/
│   ├── resnet50/
│   ├── efficientnet_b0/
│   ├── densenet121/
│   └── vgg16/
│
├── plots/
│   ├── combined_model_comparison.csv
│   ├── test_accuracy_comparison.png
│   ├── f1_score_comparison.png
│   ├── parameter_comparison.png
│   ├── training_time_comparison.png
│   ├── accuracy_vs_parameters.png
│   └── accuracy_vs_training_time.png
│
├── scripts/
│   └── run_experiment.sh
│
└── src/
    ├── main.py
    └── plot_results.py
```

---

# Dataset

Place the Tobacco3482 dataset in:

```text
data/raw/Tobacco3482/
```

Dataset source:

https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg

---

# Running Experiments

## Train all supported backbones and generate comparison plots

```bash
bash scripts/run_experiment.sh
```

## Train all supported backbones only

```bash
bash scripts/run_experiment.sh --train-only
```

## Generate comparison plots only

```bash
bash scripts/run_experiment.sh --plot-only
```

## Train selected backbones and generate plots

```bash
bash scripts/run_experiment.sh \
    --backbones resnet18 mobilenet_v2 \
    --epochs 15 \
    --folds 3
```

## Train a single backbone

### VGG16

```bash
bash scripts/run_experiment.sh --backbones vgg16
```

### ResNet50

```bash
bash scripts/run_experiment.sh --backbones resnet50
```

### DenseNet121

```bash
bash scripts/run_experiment.sh --backbones densenet121
```

> **Note (macOS):** You may optionally prefix commands with `caffeinate -dimsu` to prevent the system from sleeping during long-running experiments.

---

# Generated Outputs

## Model-specific outputs

Each trained backbone stores its results under:

```text
outputs/<model_name>/
```

Typical files include:

* `comparison.csv`
* `best_model.pth`
* `confusion_matrix.png`
* Training history and other evaluation artifacts

## Comparison plots

Running `src/plot_results.py` (or `bash scripts/run_experiment.sh`) automatically generates comparison results in:

```text
plots/
```

Example outputs:

* `combined_model_comparison.csv`
* `test_accuracy_comparison.png`
* `f1_score_comparison.png`
* `parameter_comparison.png`
* `training_time_comparison.png`
* `accuracy_vs_parameters.png`
* `accuracy_vs_training_time.png`

---

# Training Pipeline

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
outputs/<model>/comparison.csv
          │
          ▼
src/plot_results.py
          │
          ▼
plots/
```

---

# Model Performance Summary

After experiments complete, the consolidated metrics for all available models are stored in:

```text
plots/combined_model_comparison.csv
```

The generated visualizations provide comparisons of:

* Test Accuracy
* F1 Score
* Parameter Count
* Training Time
* Accuracy vs. Parameters
* Accuracy vs. Training Time

These plots are automatically created from the `comparison.csv` files generated for each trained backbone.

# Document Visual Feature Extraction

A deep learning framework for **feature extraction** and  **document image classification** using multiple CNN backbones on the **Tobacco3482** dataset. The project supports automated training, k-fold cross-validation, evaluation, model comparison, confusion matrices, and visualization of results across several state-of-the-art architectures.

---

## Project Architecture

<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/efbd0e81-4ccc-4892-bd90-e2acc3b81e25" />


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
Tobacco3482 Dataset
(ADVE, Email, Form, Letter, Memo,
 News, Note, Report, Resume, Scientific)
          │
          ▼
Dataset Loading
(Tobacco3482Dataset)
          │
          ▼
Extract Class Labels
(get_labels())
          │
          ▼
Stratified K-Fold Cross Validation
          │
          ├───────────────────────┐
          │                       │
          ▼                       ▼
     Train Split           Validation Split
          │                       │
          ▼                       ▼
 Train Transforms         Validation Transforms
 Resize(256)              Resize(224)
 RandomCrop(224)          ToTensor()
 RandomHorizontalFlip()   Normalize()
 RandomRotation(5°)
 ColorJitter()
 ToTensor()
 Normalize()
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
              DataLoaders
         (Batching & Shuffling)
                      │
                      ▼
             Backbone Selection
      (ResNet18, ResNet50, VGG16,
       DenseNet121, EfficientNet-B0,
       MobileNetV2, MobileNetV3-Small)
                      │
                      ▼
         Replace Final Classification Layer
                (Linear → 10 Classes)
                      │
                      ▼
               Training Loop
                 (Per Epoch)
                      │
                      ▼
                Forward Pass
                      │
                      ▼
             CrossEntropy Loss
                      │
                      ▼
                Backward Pass
                      │
                      ▼
            Adam Optimizer Update
                      │
                      ▼
              Validation Phase
          (Loss & Accuracy Computation)
                      │
                      ▼
               Early Stopping
      (Monitor Validation Loss)
                      │
                      ▼
            Save Best Checkpoint
          (*.pth per fold)
                      │
                      ▼
          Evaluate Best Fold Model
                      │
                      ▼
          Aggregate Predictions
            Across All Folds
                      │
                      ▼
           Performance Evaluation
      Accuracy, Precision, Recall,
      F1-Score, Confusion Matrix,
      Classification Report
                      │
                      ▼
           outputs/<model_name>/
                      │
          ├── checkpoints/
          ├── plots/
          ├── confusion_matrices/
          └── reports/
                      │
                      ▼
              comparison.csv
                      │
                      ▼
            src/plot_results.py
                      │
                      ▼
          Comparison Visualizations
      cmp_test_acc.png
      cmp_f1.png
      cmp_time.png
      cmp_params.png
```

# Training and Validation Flow :

```
Training Loop
(For Epoch = 1 ... N)
          │
          ▼
────────────────────────────────
Training Phase
────────────────────────────────
          │
          ▼
For Each Training Batch
          │
          ▼
Input Images + Labels
          │
          ▼
CNN Backbone Forward Pass
(outputs = model(imgs))
          │
          ▼
CrossEntropy Loss
(loss = criterion(outputs, labels))
          │
          ▼
Gradient Computation
(loss.backward())
          │
          ▼
Adam Parameter Update
(optimizer.step())
          │
          ▼
Update Running Loss & Accuracy
          │
          ▼
Repeat Until All Training Batches Complete
          │
          ▼
Training Epoch Metrics
(train_loss, train_acc)
          │
          ▼
────────────────────────────────
Validation Phase
────────────────────────────────
          │
          ▼
For Each Validation Batch
          │
          ▼
Forward Pass Only
(torch.no_grad())
          │
          ▼
Compute Validation Loss
          │
          ▼
Compute Validation Accuracy
          │
          ▼
Repeat Until All Validation Batches Complete
          │
          ▼
Validation Epoch Metrics
(val_loss, val_acc)
          │
          ▼
Learning Rate Scheduler Step
(CosineAnnealingLR)
          │
          ▼
Best Model Check
          │
     ┌────┴────┐
     │         │
     ▼         ▼
Improved?     Not Improved?
     │         │
     ▼         ▼
Save .pth     Increase Counter
Checkpoint
     │         │
     └────┬────┘
          │
          ▼
Early Stopping Check
          │
     ┌────┴────┐
     │         │
     ▼         ▼
Patience      Patience
Exceeded?      Not Exceeded?
     │         │
     ▼         ▼
Stop Fold    Next Epoch
Training
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

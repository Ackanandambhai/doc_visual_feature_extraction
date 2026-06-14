"""
Tobacco3482 Document Image Classification
Multi-backbone comparative study — 7 CNN backbones, k-fold cross-validation

Dataset layout (Kaggle download):
    Tobacco3482-jpg/
        ADVE/   Email/   Form/   Letter/   Memo/
        News/   Note/    Report/ Resume/   Scientific/

Usage:
    python src/train.py --dataset_root Tobacco3482-jpg --folds 5 --epochs 30
"""

import os
import time
import csv
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

NUM_CLASSES   = 10
INPUT_SIZE    = 224
EPOCHS        = 15          # override via --epochs
LEARNING_RATE = 1e-4
SEED          = 42

CLASS_NAMES = ["ADVE", "Email", "Form", "Letter", "Memo",
               "News", "Note", "Report", "Resume", "Scientific"]

# Batch sizes tuned for M1 macbook (8  GB unified RAM wiht MPS)
# Tobacco3482 is small (~3482 images) so larger batches are fine
BACKBONE_BATCH = {
    "mobilenet_v3_small": 32,
    "mobilenet_v2": 32,
    "resnet18": 32,
    "resnet50": 16,
    "efficientnet_b0": 16,
    "densenet121": 16,
    "vgg16": 8,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dataset  (folder-per-class layout)
# ─────────────────────────────────────────────

class Tobacco3482Dataset(Dataset):
    """
    Reads every image from:
        <root>/<ClassName>/<image>.jpg

    Returns (tensor, class_index).
    Class index = position in CLASS_NAMES list.
    """

    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        root = Path(root_dir)
        for idx, cls in enumerate(CLASS_NAMES):
            cls_dir = root / cls
            if not cls_dir.exists():
                log.warning("Class folder not found: %s", cls_dir)
                continue
            imgs = sorted(
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            )
            for img_path in imgs:
                self.samples.append((img_path, idx))

        log.info("Dataset: %d images across %d classes (root=%s)",
                 len(self.samples), len(CLASS_NAMES), root_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning("Cannot open %s: %s", img_path, e)
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), 128)
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_labels(self) -> list[int]:
        return [s[1] for s in self.samples]



class TransformSubset(Dataset):
    """
    Top-level wrapper to avoid multiprocessing pickling issues on macOS/MPS.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), 128)
        return self.transform(img), label


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────

def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    w = "IMAGENET1K_V1"
    if name == "resnet18":
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121(weights=w)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "vgg16":
        m = models.vgg16(weights=w)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────

# def run_epoch(model, loader, criterion, optimizer, device, train: bool):
#     model.train(train)
#     total_loss, correct, total = 0.0, 0, 0
#     ctx = torch.enable_grad() if train else torch.no_grad()
#     with ctx:
#         for imgs, labels in loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             if train:
#                 optimizer.zero_grad()
#             out = model(imgs)
#             loss = criterion(out, labels)
#             if train:
#                 loss.backward()
#                 optimizer.step()
#             total_loss += loss.item() * imgs.size(0)
#             correct    += (out.argmax(1) == labels).sum().item()
#             total      += imgs.size(0)
#     return total_loss / total, correct / total
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)

    total_loss = 0.0
    correct = 0
    total = 0

    # pbar = tqdm(
    #     loader,
    #     desc="Train" if train else "Val",
    #     leave=False,
    #     dynamic_ncols=True
    # )
    pbar = tqdm(
        loader,
        desc="Train" if train else "Val",
        leave=True,
        dynamic_ncols=True,
        ascii=False
    )
        
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, labels in pbar:

            imgs = imgs.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            avg_loss = total_loss / total
            avg_acc = correct / total

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.4f}"
            )

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────

def save_curves(history: dict, name: str, fold: int, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name}  (fold {fold}) – Training Curves", fontsize=13)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss",  color="#2563EB")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",    color="#DC2626")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc",   color="#2563EB")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",     color="#DC2626")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_fold{fold}_curves.png", dpi=150)
    plt.close()


def save_confusion(labels, preds, name: str, out_dir: Path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={"size": 9})
    ax.set_title(f"{name} – Confusion Matrix (avg over folds)", fontsize=13)
    ax.set(xlabel="Predicted", ylabel="True")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_confusion.png", dpi=150)
    plt.close()


def save_comparison_plots(results: list[dict], out_dir: Path):
    names  = [r["Model"] for r in results]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    plots  = out_dir / "plots"

    def bar_chart(values, ylabel, title, fname, fmt=".3f"):
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set(ylabel=ylabel, title=title)
        ax.set_ylim(0, max(values) * 1.15)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    format(v, fmt), ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(plots / fname, dpi=150)
        plt.close()

    bar_chart([r["Test_Accuracy"]        for r in results],
              "Test Accuracy",    "Test Accuracy Comparison",    "cmp_test_acc.png")
    bar_chart([r["F1_Score"]             for r in results],
              "F1-Score (macro)", "F1-Score Comparison",         "cmp_f1.png")
    bar_chart([r["Training_Time"]        for r in results],
              "Training Time (s)","Training Time Comparison",    "cmp_time.png", fmt=".0f")
    bar_chart([r["Parameters"] / 1e6    for r in results],
              "Parameters (M)",   "Parameter Count Comparison",  "cmp_params.png", fmt=".1f")

    log.info("Comparison plots saved.")


# ─────────────────────────────────────────────
# Per-backbone k-fold training loop
# ─────────────────────────────────────────────

def train_backbone(
    name: str,
    full_dataset: Tobacco3482Dataset,
    n_folds: int,
    epochs: int,
    out_dir: Path,
    device: torch.device,
) -> dict:
    log.info("=" * 60)
    log.info("  Backbone: %s  (%d-fold CV, %d epochs/fold)", name, n_folds, epochs)
    log.info("=" * 60)

    batch_size = BACKBONE_BATCH.get(name, 32)
    nw         = 0 if device.type == "mps" else min(4, os.cpu_count())
    params     = count_parameters(build_model(name))
    log.info("Parameters: {:,}  |  batch_size: {}".format(params, batch_size))

    all_labels_agg, all_preds_agg = [], []
    fold_train_accs, fold_val_accs = [], []
    t_start = time.time()

    labels_arr = np.array(full_dataset.get_labels())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_dataset)), labels_arr), 1):
        log.info("--- Fold %d / %d ---", fold, n_folds)

        # Build per-fold datasets with correct transforms
        train_ds = Subset(full_dataset, train_idx)
        val_ds   = Subset(full_dataset, val_idx)

        # Apply different transforms per fold split

        train_loader = DataLoader(
            TransformSubset(train_ds, get_transforms(True)),
            batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(
            TransformSubset(val_ds, get_transforms(False)),
            batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))

        model     = build_model(name).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Early stopping configuration
        # We monitor validation loss instead of validation accuracy
        # because it is generally a more stable convergence signal.patience=3:
        # Stop training if validation loss does not improve for 3 consecutive epochs.

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_acc = 0.0
        best_val_loss = float("inf")

        # Number of consecutive epochs without improvement.
        epochs_without_improvement = 0

        # Maximum tolerated epochs without improvement.
        EARLY_STOPPING_PATIENCE = 3

        best_ckpt = out_dir / "checkpoints" / f"{name}_fold{fold}_best.pth"

        # for ep in range(1, epochs + 1):
        #     t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        #     v_loss, v_acc = run_epoch(model, val_loader,   criterion, None,      device, False)
        #     scheduler.step()

        #     history["train_loss"].append(t_loss)
        #     history["val_loss"].append(v_loss)
        #     history["train_acc"].append(t_acc)
        #     history["val_acc"].append(v_acc)

        #     log.info("  Ep %2d/%d  tl=%.4f vl=%.4f  ta=%.4f va=%.4f",
        #              ep, epochs, t_loss, v_loss, t_acc, v_acc)

        #     if v_acc > best_val_acc:
        #         best_val_acc = v_acc
        #         torch.save(model.state_dict(), best_ckpt)
        for ep in range(1, epochs + 1):

            print("\n" + "=" * 60)
            print(f"Epoch {ep}/{epochs}")
            print("=" * 60)

            t_loss, t_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, True
            )

            v_loss, v_acc = run_epoch(
                model, val_loader, criterion, None, device, False
            )

            scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            log.info(
                "  Ep %2d/%d  tl=%.4f vl=%.4f  ta=%.4f va=%.4f",
                ep, epochs, t_loss, v_loss, t_acc, v_acc
            )

            # Save checkpoint only when validation loss improves.
            if v_loss < best_val_loss:

                best_val_loss = v_loss
                best_val_acc = v_acc

                epochs_without_improvement = 0

                torch.save(model.state_dict(), best_ckpt)

            else:

                epochs_without_improvement += 1

            # Early stopping: terminate fold training when validation loss has not
            # improved for PATIENCE consecutive epochs.
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

                log.info(
                    "Early stopping triggered at epoch %d "
                    "(best validation loss = %.4f)",
                    ep,
                    best_val_loss,
                )

                break

        fold_train_accs.append(history["train_acc"][-1])
        fold_val_accs.append(best_val_acc)
        log.info("  Fold %d best val_acc: %.4f", fold, best_val_acc)

        # Save curves for this fold
        save_curves(history, name, fold, out_dir / "plots")

        # Evaluate best model on validation fold → aggregate predictions
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        lbls, preds = evaluate(model, val_loader, device)
        all_labels_agg.extend(lbls)
        all_preds_agg.extend(preds)

    training_time = time.time() - t_start
    log.info("Total training time: %.1f s", training_time)

    all_labels_agg = np.array(all_labels_agg)
    all_preds_agg  = np.array(all_preds_agg)

    test_acc = (all_labels_agg == all_preds_agg).mean()
    prec     = precision_score(all_labels_agg, all_preds_agg, average="macro", zero_division=0)
    rec      = recall_score(all_labels_agg,    all_preds_agg, average="macro", zero_division=0)
    f1       = f1_score(all_labels_agg,        all_preds_agg, average="macro", zero_division=0)

    save_confusion(all_labels_agg, all_preds_agg, name, out_dir / "confusion_matrices")

    report = classification_report(
        all_labels_agg, all_preds_agg, target_names=CLASS_NAMES, zero_division=0)
    (out_dir / "reports" / f"{name}_report.txt").write_text(report)
    log.info("\n%s", report)

    return {
        "Model":          name,
        "Parameters":     params,
        "Train_Accuracy": round(float(np.mean(fold_train_accs)), 4),
        "Val_Accuracy":   round(float(np.mean(fold_val_accs)), 4),
        "Test_Accuracy":  round(float(test_acc), 4),
        "Precision":      round(float(prec), 4),
        "Recall":         round(float(rec), 4),
        "F1_Score":       round(float(f1), 4),
        "Training_Time":  round(training_time, 1),
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tobacco3482 multi-backbone training")
    parser.add_argument("--dataset_root", default=".",
                        help="Folder containing ADVE/, Email/, … sub-directories")
    parser.add_argument("--out_dir",   default="outputs")
    parser.add_argument("--epochs",    type=int, default=EPOCHS)
    parser.add_argument("--folds",     type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONE_BATCH.keys()))
    parser.add_argument("--seed",      type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s  (%.1f GB)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    out_dir = Path(args.out_dir)
    for sub in ("checkpoints", "plots", "confusion_matrices", "reports"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Load entire dataset once (transforms applied per split inside loop)
    full_dataset = Tobacco3482Dataset(args.dataset_root, transform=None)

    results = []
    for name in args.backbones:
        result = train_backbone(
            name=name,
            full_dataset=full_dataset,
            n_folds=args.folds,
            epochs=args.epochs,
            out_dir=out_dir,
            device=device,
        )
        results.append(result)

    csv_path   = out_dir / "comparison.csv"
    fieldnames = ["Model","Parameters","Train_Accuracy","Val_Accuracy",
                  "Test_Accuracy","Precision","Recall","F1_Score","Training_Time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    log.info("Results written to %s", csv_path)

    save_comparison_plots(results, out_dir)
    log.info("All done.")


if __name__ == "__main__":
    main()

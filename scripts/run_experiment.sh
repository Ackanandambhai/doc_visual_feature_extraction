#!/usr/bin/env bash
# run_experiment.sh  -  Train all 7 backbones on Tobacco3482 with k-fold CV
#
# Usage:
#   bash run_experiment.sh
#   bash run_experiment.sh --epochs 50 --folds 5
#   bash run_experiment.sh --backbones resnet18 mobilenet_v2

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-Tobacco3482-jpg}"
OUT_DIR="${OUT_DIR:-outputs}"
EPOCHS="${EPOCHS:-30}"
FOLDS="${FOLDS:-5}"

echo "=================================================="
echo "  Tobacco3482 Backbone Comparison Experiment"
echo "  Dataset : $DATASET_ROOT"
echo "  Output  : $OUT_DIR"
echo "  Epochs  : $EPOCHS per fold"
echo "  Folds   : $FOLDS"
echo "=================================================="

python src/train.py \
    --dataset_root "$DATASET_ROOT" \
    --out_dir      "$OUT_DIR"      \
    --epochs       "$EPOCHS"       \
    --folds        "$FOLDS"        \
    "$@"

echo ""
echo "Re-generating comparison plots..."
python src/plot_results.py --csv "$OUT_DIR/comparison.csv" --out_dir "$OUT_DIR"
echo ""
echo "All outputs saved to: $OUT_DIR/"
echo "Summary CSV: $OUT_DIR/comparison.csv"

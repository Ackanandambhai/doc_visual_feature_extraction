#!/usr/bin/env bash
# run_experiment.sh  -  Train all 7 backbones on Tobacco3482 with k-fold CV
#
# Usage:
#   bash run_experiment.sh
#   bash run_experiment.sh --epochs 15 --folds 3
#   bash run_experiment.sh --backbones resnet18 mobilenet_v2

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/raw/Tobacco3482}"
OUT_DIR="${OUT_DIR:-outputs}"
EPOCHS="${EPOCHS:-15}"
FOLDS="${FOLDS:-3}"

echo "=================================================="
echo "  Tobacco3482 Backbone Comparison Experiment"
echo "  Dataset : $DATASET_ROOT"
echo "  Output  : $OUT_DIR"
echo "  Epochs  : $EPOCHS per fold"
echo "  Folds   : $FOLDS"
echo "=================================================="

DEFAULT_BACKBONES=(
    mobilenet_v3_small
    mobilenet_v2
    resnet18
    resnet50
    efficientnet_b0
    densenet121
    vgg16
)

BACKBONES=("${DEFAULT_BACKBONES[@]}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbones)
            BACKBONES=("$2")
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --folds)
            FOLDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

for MODEL in "${BACKBONES[@]}"
do
    echo ""
    echo "=================================================="
    echo "Training $MODEL"
    echo "=================================================="

    python src/main.py \
        --dataset_root "$DATASET_ROOT" \
        --out_dir "$OUT_DIR/$MODEL" \
        --epochs "$EPOCHS" \
        --folds "$FOLDS" \
        --backbones "$MODEL"

done

echo ""
echo "All experiments completed."

#!/usr/bin/env bash
#
# run_experiment.sh
#
# Usage:
#
#   # Train all models + generate plots (default)
#   bash scripts/run_experiment.sh
#
#   # Train only
#   bash scripts/run_experiment.sh --train-only
#
#   # Generate plots only
#   bash scripts/run_experiment.sh --plot-only
#
#   # Train selected models then generate plots
#   bash scripts/run_experiment.sh \
#       --backbones resnet18 mobilenet_v2 \
#       --epochs 15 \
#       --folds 3
#

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/raw/Tobacco3482}"
OUT_DIR="${OUT_DIR:-outputs}"
EPOCHS="${EPOCHS:-15}"
FOLDS="${FOLDS:-3}"

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

RUN_TRAIN=true
RUN_PLOT=true

# -----------------------------
# Parse arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in

        --train-only)
            RUN_TRAIN=true
            RUN_PLOT=false
            shift
            ;;

        --plot-only)
            RUN_TRAIN=false
            RUN_PLOT=true
            shift
            ;;

        --backbones)
            shift
            BACKBONES=()

            while [[ $# -gt 0 && "$1" != --* ]]; do
                BACKBONES+=("$1")
                shift
            done
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
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Dataset : $DATASET_ROOT"
echo "Outputs : $OUT_DIR"
echo "Epochs  : $EPOCHS"
echo "Folds   : $FOLDS"
echo "=============================================="

# -----------------------------
# Train models
# -----------------------------
if [ "$RUN_TRAIN" = true ]; then

    for MODEL in "${BACKBONES[@]}"
    do
        echo ""
        echo "=============================================="
        echo "Training: $MODEL"
        echo "=============================================="

        python src/main.py \
            --dataset_root "$DATASET_ROOT" \
            --out_dir "$OUT_DIR/$MODEL" \
            --epochs "$EPOCHS" \
            --folds "$FOLDS" \
            --backbones "$MODEL"

    done

    echo ""
    echo "Training completed."
fi

# -----------------------------
# Generate plots
# -----------------------------
if [ "$RUN_PLOT" = true ]; then

    echo ""
    echo "=============================================="
    echo "Generating comparison plots..."
    echo "=============================================="

    python src/plot_results.py

    echo ""
    echo "Plots generated successfully."
fi

echo ""
echo "=============================================="
echo "Pipeline completed successfully."
echo "=============================================="
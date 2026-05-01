#!/bin/bash

MODELS=(
    "resnet18" "resnet50" "resnet101" "resnet152"
    "vgg11" "vgg13" "vgg16" "vgg19"
    "convnext_tiny" "convnext_small" "convnext_base"
    "convnext_large"
)
VARIANTS=("" "-mask" "-mask-2" "-mask-3" "-mask-4" "-mask-5")

REX_SCRIPT_PATH=""
BASE_RESULTS_DIR="/mnt/data/Documents/adversarial-attacks-pytorch"

for MODEL in "${MODELS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        model=${MODEL//_/-}
        ATTACK_FOLDER="${BASE_RESULTS_DIR}/${model}/${model}-one-pixel${VARIANT}"
        REX_SCRIPT_PATH="/mnt/data/Documents/ReX/scripts/pytorch_${MODEL}.py"
        # Only run if the folder actually exists
        if [ -d "$ATTACK_FOLDER" ]; then
            echo "====================================================="
            echo "Processing: Model -> $MODEL | Variant -> $VARIANT"
            echo "Folder: $ATTACK_FOLDER"
            echo "====================================================="

            python experiments/run_rex_attack.py \
                --attack_folder "$ATTACK_FOLDER" \
                --model "$MODEL" \
                --script "$REX_SCRIPT_PATH"
        else
            echo "Skipping: $ATTACK_FOLDER (Directory not found)"
        fi

    done
done

echo "All extractions complete!"
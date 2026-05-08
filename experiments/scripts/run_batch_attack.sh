#!/bin/bash

MODELS_RESNET=("resnet18" "resnet50" "resnet101" "resnet152")
MODELS_VGG=("vgg11" "vgg13" "vgg16" "vgg19")
MODELS_CONVNEXT=("convnext_tiny" "convnext_small" "convnext_base" "convnext_large")


ATTACKS=(
  "onepixel"
  #"pixle"
)

BASE_FOLDER="/mnt/data/Documents/adversarial-attacks-pytorch"
BASE_XAI_FOLDER="/mnt/data/Documents/ReX"
DATASET_PATH="/mnt/data/Documents/imagenet-1k"

# Default experiment parameters
DEFAULT_NUM_SAMPLES=100
DEFAULT_BATCH_SIZE=32
DEFAULT_SEED=0
DEFAULT_DEVICE="cuda"  # or "cpu"

for MODEL in "${MODELS_RESNET[@]}" "${MODELS_VGG[@]}" "${MODELS_CONVNEXT[@]}"; do
    for ATTACK in "${ATTACKS[@]}"; do
        ATTACK_FOLDER="${BASE_FOLDER}/${MODEL}/${MODEL}-${ATTACK}"
        JSON_FILE="${ATTACK_FOLDER}/results_${ATTACK}_${MODEL}.json"
        XAI_FOLDER="${BASE_XAI_FOLDER}/${MODEL}_exp"
        OUTPUT_FOLDER="${ATTACK_FOLDER}/analysis"

        if [ -d "$ATTACK_FOLDER" ]; then
            echo "Running attack: ${ATTACK} for model: ${MODEL}"
            python experiments/runner.py \
                --model "$MODEL" \
                --attack "$ATTACK" \
                --dataset_path "$DATASET_PATH" \
                --num_samples "$DEFAULT_NUM_SAMPLES" \
                --batch_size "$DEFAULT_BATCH_SIZE" \
                --seed "$DEFAULT_SEED" \
                --device "$DEFAULT_DEVICE"

            echo "Running analysis for attack: ${ATTACK} and model: ${MODEL}"
            python experiments/analysis.py \
                --json_file "$JSON_FILE" \
                --results_dir "$ATTACK_FOLDER" \
                --xai_results "$XAI_FOLDER" \
                --output_dir "$OUTPUT_FOLDER"
        else
            echo "Attack folder not found for model: ${MODEL} and attack: ${ATTACK}"
        fi
    done
done
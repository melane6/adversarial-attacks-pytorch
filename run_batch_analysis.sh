#!/bin/bash

MODELS=(
    "resnet18" "resnet50" "resnet101" "resnet152"
)

varients=(
    "" "-mask" "-mask-2" "-mask-3" "-mask-4" "-mask-5"
)
BASE_FOLDER="/mnt/data/Documents/adversarial-attacks-pytorch"
BASE_XAI_FOLDER="/mnt/data/Documents/ReX"

# define in loop
JSON_FILE=""
ATTACK_FOLDER=""
XAI_FOLDER=""


for MODEL in "${MODELS[@]}"; do
    for VARIANT in "${varients[@]}"; do
        model=${MODEL//_/-}
        ATTACK_FOLDER="${BASE_FOLDER}/${model}/${model}-one-pixel${VARIANT}"
        JSON_FILE="${ATTACK_FOLDER}/results_onepixel_${MODEL}.json"
        XAI_FOLDER="${BASE_XAI_FOLDER}/${MODEL}_exp"
        mask_keys=("explanation_0" "explanation_1" "explanation_2" "explanation_3" "explanation_4")
        OUTPUT_FOLDER="${ATTACK_FOLDER}/analysis${VARIANT}"
        if [ "$VARIANT" == "" ]; then
            for key in "${mask_keys[@]}"; do
                if [ -d "$ATTACK_FOLDER" ]; then
                    echo "${ATTACK_FOLDER} with variant: ${VARIANT} and mask key: ${key}"
                    python experiments/analysis.py --json_file "$JSON_FILE" --results_dir "$ATTACK_FOLDER" --xai_results "$XAI_FOLDER" --mask_keys "$key" --output_dir "$OUTPUT_FOLDER"
                fi
            done
        fi
        echo "Running analysis for model: ${MODEL} with variant: ${VARIANT}"
    done
done
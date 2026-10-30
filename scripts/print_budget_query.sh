#!/bin/bash

# Budget query for all models and variants.
POPSIZE=(
    10 20 30 40 50 60 
)
STEPS=(
    10 20 30 40 50 60 70 80 100 120 150 200 300 400 500
)

MODELS=(
    "resnet18"
     "resnet50" "resnet101" "resnet152"
    "vgg11" "vgg13" "vgg16" "vgg19"
     "convnext_tiny" "convnext_small" "convnext_base"
     "convnext_large"
)
VARIANTS=("" "-mask" "-mask-3" "-mask-5")
BASE_FOLDER="/mnt/data/Documents/one-pixel-results"
BASE_XAI_FOLDER="/mnt/data/Documents/ReX"
DATASET_PATH="/mnt/data/Documents/ImageNet-Mini/"
# Default experiment parameters
DEFAULT_NUM_SAMPLES=1000
DEFAULT_BATCH_SIZE=32
DEFAULT_SEED=0
results=()
ATTACK="onepixel"


OUTPUT_CSV="${BASE_FOLDER}/budget_query_results_$(date +%Y%m%d_%H%M%S).csv"
echo "Model,Variant,PopSize,Steps,ASR" > "$OUTPUT_CSV"

for MODEL in "${MODELS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        model=${MODEL//_/-}
        ATTACK_FOLDER="${BASE_FOLDER}/${model}/${model}-one-pixel${VARIANT}"
        XAI_FOLDER="${BASE_XAI_FOLDER}/${MODEL}_exp"
        echo "Running budget query for model: ${MODEL} with variant: ${VARIANT}"
        for POP in "${POPSIZE[@]}"; do
            for STEP in "${STEPS[@]}"; do
                if [ ! -d "$ATTACK_FOLDER" ]; then
                    echo "Skipping: $ATTACK_FOLDER (Directory not found)"
                    continue
                fi
                folder=""
		if [[ "$VARIANT" == *"-mask"* ]]; then
                    folder="$XAI_FOLDER"
                    if [ ! -d "$XAI_FOLDER" ]; then
                      echo "Skipping: $XAI_FOLDER (Directory not found)"
                      continue
                    fi
                fi
                OUTPUT_FOLDER="${ATTACK_FOLDER}/budget_query${VARIANT}-popsize${POP}-steps${STEP}"
                echo "Model: ${MODEL}, Variant: ${VARIANT}, Population Size: ${POP}, Steps: ${STEP}"
                echo "Saving results to: ${OUTPUT_FOLDER}"
                if [ -f "${OUTPUT_FOLDER}/results_onepixel_${MODEL}.json" ]; then
		    ASR=$( grep "attack_success_rate" "${OUTPUT_FOLDER}/results_onepixel_${MODEL}.json" | sed 's/.*: //' | sed 's/[",]//g' )
                    result_entry="${MODEL},${VARIANT:-none},${POP},${STEP},${ASR}"
                    results+=("$result_entry")
                    echo "$result_entry" >> "$OUTPUT_CSV"
                    echo "Added result: $result_entry"
                else
                    echo "Warning: Results file not found at ${OUTPUT_FOLDER}/results_onepixel_${MODEL}.json"
                fi
            done
        done
    done
done

echo "All budget queries complete! Here are the results:"
for result in "${results[@]}"; do
    echo "$result"
done

echo "Results saved to: $OUTPUT_CSV"
echo "CSV Preview:"
head -n 20 "$OUTPUT_CSV"

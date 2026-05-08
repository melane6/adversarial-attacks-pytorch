#!/bin/bash

# Budget query for all models and variants.
POPSIZE=(
    10 60 
)
STEPS=(
    10 20 40 80 100 150 200 400
)

MODELS=(
    #"resnet18"
    #"resnet50" "resnet101" "resnet152"
   # "vgg11"
    "vgg13" "vgg16" "vgg19"
    # "convnext_tiny" "convnext_small" "convnext_base"
    # "convnext_large"
)
VARIANTS=("" "-mask" "-mask-3" "-mask-5")
BASE_FOLDER="/mnt/data/Documents/adversarial-attacks-pytorch"
BASE_XAI_FOLDER="/mnt/data/Documents/ReX"
DATASET_PATH="/mnt/data/Documents/ImageNet-Mini/"
# Default experiment parameters
DEFAULT_NUM_SAMPLES=1000
DEFAULT_BATCH_SIZE=32
DEFAULT_SEED=0
results=()
ATTACK="onepixel"

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

                python_args=(
                    "experiments/runner.py"
                    "--model" "$MODEL"
                    "--attack" "$ATTACK"
                    "--dataset-path" "$DATASET_PATH"
                    "--num-samples" "$DEFAULT_NUM_SAMPLES"
                    "--batch-size" "$DEFAULT_BATCH_SIZE"
                    "--seed" "$DEFAULT_SEED"
                    "--popsize" "$POP"
                    "--steps" "$STEP"
                    "--output-dir" "$OUTPUT_FOLDER"
                )

                if [ -n "$folder" ]; then
                    python_args+=("--mask-folder" "$folder")
		    if [[ "$VARIANT" =~ ([0-9]+) ]]; then
			number="${BASH_REMATCH[1]}"
			python_args+=("--num-exp" "$number")
		    fi
                fi
                echo "Running command: python ${python_args[*]}"
                python "${python_args[@]}"

                if [ -f "${OUTPUT_FOLDER}/results_onepixel_${MODEL}.json" ]; then
                    ASR=$( grep "attack_success_rate" "${OUTPUT_FOLDER}/results_onepixel_${MODEL}.json" )
                    results+=("Model: ${MODEL}, Variant: ${VARIANT}, Population Size: ${POP}, Steps: ${STEP}, ASR: ${ASR}")
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

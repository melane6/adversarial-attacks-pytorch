#!/bin/bash

# Budget query for all models and variants.
POPSIZE=(
    10 
    #30 60 
)
STEPS=(
    10 
    #20 40 80 100 
)

MODELS=(
    #"resnet18"
    #"resnet50" 
    #"resnet101" 
    #"resnet152"
    #"vgg11"
    #"vgg13" 
    #"vgg16"
    "vgg19"
     #"convnext_tiny"
     #"convnext_small" 
     #"convnext-base"
     #"convnext-large"
)
VARIANTS=(
    "" ) 
    #"-mask-5" 
    #"gradient_shap"
    #"guided_grad_cam"
    #"integrated_gradients"
    #"layer_grad_cam")
	#"-mask")
       	#"-necessity")


BASE_FOLDER="/mnt/data/Documents/one-pixel-results"
BASE_XAI_FOLDER="/mnt/data/Documents/ReX/"
DATASET_PATH="/mnt/data/Documents/ImageNet-Mini/"
# Default experiment parameters
DEFAULT_NUM_SAMPLES=1000
DEFAULT_BATCH_SIZE=64
DEFAULT_SEED=0
results=()
ATTACK="onepixel"
COMPLETE_EXP=false
USE_COLOR=false
exp_key="explanation"
use_other_xai=false
override_num_mask=1

for MODEL in "${MODELS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        model=${MODEL//_/-}
        ATTACK_FOLDER="${BASE_FOLDER}/${model}/"
        if [[ $use_other_xai == true ]]; then
            XAI_FOLDER="${ATTACK_FOLDER}/${MODEL}-one-pixel/budget_query-popsize10-steps10/${MODEL}_${VARIANT}_exp"
        else
            XAI_FOLDER="${BASE_XAI_FOLDER}/${MODEL}_exp"
        fi
        
        echo "Running budget query for model: ${MODEL} with variant: ${VARIANT}"
        for POP in "${POPSIZE[@]}"; do
            for STEP in "${STEPS[@]}"; do
                if [ ! -d "$ATTACK_FOLDER" ]; then
                    echo "Skipping: $ATTACK_FOLDER (Directory not found)"
                    continue
                fi
                folder=""
		        if [[ "$VARIANT" != "" ]]; then
                    folder="$XAI_FOLDER"
		            COMPLETE_EXP=false
		            if [ ! -d "$XAI_FOLDER" ]; then
                      echo "Skipping: $XAI_FOLDER (Directory not found)"
                      continue
                    fi
                fi
                OUTPUT_FOLDER="${ATTACK_FOLDER}/${model}${VARIANT}-popsize${POP}-steps${STEP}"
                echo "Model: ${MODEL}, Variant: ${VARIANT}, Population Size: ${POP}, Steps: ${STEP}"
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
                    "--exp_key" "$exp_key"
                )
		        if [ "$COMPLETE_EXP" = true ] && [[ "$VARIANT" == *"-necessity"* ]]; then
                    python_args+=("--complete-exp" "True")
                    folder="${BASE_XAI_FOLDER}/${MODEL}_complete_exp"
                fi

                if [ -n "$folder" ]; then
                    python_args+=("--mask-folder" "$folder")
                    if [ $override_num_mask -gt 0 ]; then
                        number=$override_num_mask
                         python_args+=("--num-exp" "$number")
		            elif [[ "$VARIANT" =~ ([0-9]+) ]]; then
			            number="${BASH_REMATCH[1]}"
                         python_args+=("--num-exp" "$number")
		            fi
                fi
		        if [ "$USE_COLOR" = true ]; then
                    python_args+=("--colour" 0 0 0)
		            OUTPUT_FOLDER+="-color"
                fi
		        python_args+=("--output-dir" "$OUTPUT_FOLDER")
		        echo "Saving results to: ${OUTPUT_FOLDER}"
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

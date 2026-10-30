#!/bin/bash

MODELS=(
#    "resnet18"
     "resnet50" 
#     "resnet101" 
#     "resnet152"
#    "vgg11"
 #   "vgg13" 
 #  "vgg16" 
 #  "vgg19"
#    "convnext_tiny" "convnext_small" "convnext_base"
#    "convnext_large"
)
VARIANTS=("")
#"-mask" "-mask-2" "-mask-3" "-mask-4" "-mask-5")
XAI_METHODS=(
#	"layer_grad_cam") 
"integrated_gradients")
#"gradient_shap")
# "guided_grad_cam" )

BASE_RESULTS_DIR="/mnt/data/Documents/one-pixel-results/"

for MODEL in "${MODELS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        for method in "${XAI_METHODS[@]}"; do
            model=${MODEL//_/-}
            ATTACK_FOLDER="${BASE_RESULTS_DIR}/${model}/${model}-one-pixel${VARIANT}/budget_query-popsize10-steps10/"
            REX_SCRIPT_PATH="/mnt/data/Documents/ReX/scripts/pytorch_${MODEL}.py"
            # Only run if the folder actually exists
            if [ -d "$ATTACK_FOLDER" ]; then
                echo "====================================================="
                echo "Processing: Model -> $MODEL | Variant -> $VARIANT"
                echo "Folder: $ATTACK_FOLDER"
                echo "XAI Method: $method"
                echo "====================================================="

                python experiments/run_xai.py \
                    --attack_folder "$ATTACK_FOLDER" \
                    --model "$MODEL" \
                    --script "$REX_SCRIPT_PATH" \
                    --xai_method "$method"
            else
                echo "Skipping: $ATTACK_FOLDER (Directory not found)"
            fi
        done
    done
done

echo "All extractions complete!"

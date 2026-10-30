#!/bin/bash

# Takes user inputs and runs attack
# Inputs:
#   -m: model name (e.g., resnet18, vgg16, convnext_small)
#   -a: attack type (e.g., onepixel, pixle)
#   -p: number of pixels to perturb (default: 1)
#   -n: number of samples to attack (default: 1000)
#   -d: dataset path (default: /mnt/data/Documents/ImageNet-Mini/)
#   -pop: population size for the attack (default: 10)
#   -s: number of steps for the attack (default: 10)

BATCH_SIZE=64
EXP_KEY="explanation"

model_name=${1:-"resnet18"}
num_pixels=${2:-1}
num_samples=${3:-1000}
pop_size=${4:-10}
num_steps=${5:-10}
color=${6:-""}
output_dir=${7:-"/mnt/data/Documents/one-pixel-results"}
seed=${8:-0}
dataset_path=${9:-"/mnt/data/Documents/ImageNet-Mini/"}
attack_type=${10:-"onepixel"}



output_folder="${output_dir}/${model_name}/${model_name}-${attack_type}-popsize${pop_size}-steps${num_steps}"
if 

python_args=(
    "experiments/runner.py"
    "--model" "$model_name"
    "--attack" "$attack_type"
    "--pixels" "$num_pixels"
    "--num-samples" "$num_samples"
    "--dataset-path" "$dataset_path"
    "--batch-size" "$BATCH_SIZE"
    "--popsize" "$pop_size"
    "--steps" "$num_steps"
    "--exp_key" "$EXP_KEY"
    "--output-dir" "$output_folder"
    "--seed" "$seed"
)
if [ "$color" != "" ]; then
    python_args+=("--colour" "$color")
fi

echo "Running attack with the following parameters:"
echo "Model: $model_name"
echo "Attack Type: $attack_type"    
echo "Number of Pixels: $num_pixels"
echo "Number of Samples: $num_samples"
echo "Population Size: $pop_size"
echo "Number of Steps: $num_steps"
echo "Dataset Path: $dataset_path"
echo "Output Directory: $output_folder"

python "${python_args[@]}"

echo "Attack completed. Results saved in: $output_folder"
if [ -f "${output_folder}/results_onepixel_${model_name}.json" ]; then
    ASR=$( grep "attack_success_rate" "${output_folder}/results_onepixel_${model_name}.json" )
    results+=("Model: ${model_name}, Variant: ${variant}, Population Size: ${pop_size}, Steps: ${num_steps}, ASR: ${ASR}")
else
    echo "Warning: Results file not found at ${output_folder}/results_onepixel_${model_name}.json"
fi
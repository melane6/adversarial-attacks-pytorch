import os
import glob
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
import logging
import importlib
from typing import List
import torch.nn.functional as F


import torch
from rex_xai.explanation.explanation import Explanation
from rex_xai.input.config import CausalArgs
from rex_xai.responsibility.resp_maps import ResponsibilityMaps
from rex_xai.responsibility.prediction import Prediction, Predictions
from rex_xai.utils._utils import Strategy

logging.basicConfig(level=logging.INFO)
def from_pytorch_tensor(tensor, target=None) -> Predictions | List[Predictions]:
    softmax_tensor = F.softmax(tensor, dim=1)
    prediction_scores, pred_labels = torch.topk(softmax_tensor, 1)
    prediction: List[Prediction] = []
    batch_size = tensor.shape[0]
    if batch_size == 1:
        for i, (ps, pl) in enumerate(zip(prediction_scores, pred_labels)):
            p = Prediction(pl.item(), ps.item())
            if target is not None:
                p.target = target
                p.target_confidence = softmax_tensor[i, target[0].classification].item()
            prediction.append(p)
        return Predictions(prediction)
    else:
        # more than one batch
        predictions: List[Predictions] = []
        for i in range(batch_size):
            batch_pred = []
            p = Prediction(pred_labels[i].item(), prediction_scores[i].item())
            if target is not None:
                p.target = target
                p.target_confidence = softmax_tensor[i, target[0].classification].item()
            batch_pred.append(p)
            predictions.append(Predictions(batch_pred))
        return predictions

def default_prediction_function(model):
    def inner(mutants, target=None, raw=False):
        with torch.no_grad():
            tensor = model(mutants)
            if raw:
                return F.softmax(tensor, dim=1)
            return from_pytorch_tensor(tensor, target=target)
    return inner

def get_pred_function(rex_script: str):
    name, _ = os.path.splitext(rex_script)
    spec = importlib.util.spec_from_file_location(name, rex_script)
    script = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(script)  # type: ignore
    except Exception as e:
        print(f"Error loading ReX script {rex_script}: {e}")
        return None, None
    if hasattr(script, "prediction_function"):
        pred_func = script.prediction_function
    else:
        print(f"Loading ReX script from {rex_script}: functions in script: {dir(script)}")
        pred_func = default_prediction_function(script.model)

    if hasattr(script, "preprocess"):
        preprocess = script.preprocess
    else:
        preprocess = None
        print(f"No preprocess function found in {rex_script}")

    return pred_func, preprocess

def extract_explanation(image_path: str, output_dir: str, heatmap: np.ndarray, rex_script: str):
    pred_func, preprocess = get_pred_function(rex_script)
    data = preprocess(image_path, ("N", 3, 224, 224), device=torch.device("cuda"))
    target = pred_func(data.data)
    resp_map = ResponsibilityMaps(style="Multiplicative", height=heatmap.shape[0], width=heatmap.shape[1])
    print(target[0].classification)
    print(f"Data: {data.data.shape}")
    resp_map.new_map(target[0].classification)
    resp_map.maps[target[0].classification] = heatmap.squeeze(0).squeeze(0).astype("float32")
    print(heatmap.squeeze(0).shape)
    data.target = target[0]
    data.targets = target
    data.device = "cuda"
    data.mask_value = 0
    causal_args = CausalArgs()
    causal_args.seed = 0
    causal_args.minimum_confidence_threshold = 0.5
    causal_args.gpu = True
    causal_args.strategy = Strategy.Global
    causal_args.mask_value = 0
    exp = Explanation(
        resp_map,
        pred_func,
        data,
        causal_args,
        {},
    )
    exp.extract()
    mask = exp.sufficiency_mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask = mask
    base_name = os.path.basename(image_path)
    np.save(os.path.join(output_dir, f"{base_name.split(".")[0]}_rex_explanation.npy"), mask)
    exp.save(os.path.join(output_dir, f"{base_name.split(".")[0]}_rex_explanation.png"), exp.sufficiency_mask)
    return mask, exp.sufficiency_confidence




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_folder", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet152")
    parser.add_argument("--script", type=str, default="")
    args = parser.parse_args()

    attack_info_files = glob.glob(os.path.join(args.attack_folder, "*.json"))
    if not attack_info_files:
        print(f"No JSON files found in {args.attack_folder}. Skipping.")
        return

    attack_info = attack_info_files[0]
    with open(attack_info, 'r') as f:
        results = json.load(f)

    samples = pd.json_normalize(results.get('samples', []))

    OUTPUT_DIR = os.path.join(args.attack_folder, f"{args.model}_exp")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DUMP_PATH = os.path.join(OUTPUT_DIR, f"{args.model}_attacked_rex.csv")
    # Run on all the Successful attacks
    for idx, sample in samples.iterrows():
        if not sample.get('attack_success', False):
            continue

        image_name = str(sample['image_name']).replace('.JPEG', '')
        class_name = sample['class_name']
        pred_clean = sample['pred_clean']
        pred_adv = sample['pred_adv']
        image_path = sample['image_path']

        output_path = os.path.join(OUTPUT_DIR, f"{image_name}_rex.jpeg")
        if os.path.exists(output_path):
            print(f"ReX output already exists for {image_name}, skipping...")
            continue
        print(f"[{args.model}] Image: {image_name} | Clean: {pred_clean} -> Adv: {pred_adv}")
        try:
            subprocess.run([
                "ReX", image_path,
                "--script", args.script,
                "-v",
                "--dump", DUMP_PATH,
                "--config", "/mnt/data/Documents/ReX/rex.toml",
                "--custom_target", str(pred_adv),
                "--no_extract",
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ReX failed on {image_name}: {e}")


if __name__ == "__main__":
    main()
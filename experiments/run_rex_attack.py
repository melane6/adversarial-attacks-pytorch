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
import tqdm


import torch
from rex_xai.explanation.multi_explanation import MultiExplanation
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
    resp_map.maps[target[0].classification] = heatmap[0].astype("float32")
    assert resp_map.maps[target[0].classification].shape == (224, 224), f"Responsibility map shape {resp_map.maps[target[0].classification].shape} does not match expected shape (224, 224)"
    data.target = target[0]
    data.targets = target
    data.device = "cuda"
    data.mask_value = 0
    causal_args = CausalArgs()
    causal_args.seed = 0
    causal_args.minimum_confidence_threshold = 0.5
    causal_args.gpu = True
    causal_args.strategy = Strategy.MultiSpotlight
    causal_args.mask_value = 0
    causal_args.spotlights = 5
    exp = MultiExplanation(
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
    for i, multi_exp in enumerate(exp.explanations):
        if isinstance(multi_exp, torch.Tensor):
            multi_exp = multi_exp.detach().cpu().numpy()
        elif isinstance(multi_exp, np.ndarray):
            multi_exp = multi_exp
        np.save(os.path.join(output_dir, f"{base_name.split(".")[0]}_rex_explanation_{i}.npy"), multi_exp)

    causal_args.strategy = Strategy.Contrastive
    causal_args.complete = True
    exp = Explanation(
        resp_map,
        pred_func,
        data,
        causal_args,
        {},
    )
    exp.extract()
    if exp.necessity_mask is not None:
        mask = exp.necessity_mask
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask = mask
        np.save(os.path.join(output_dir, f"{base_name.split(".")[0]}_rex_necessity_explanation.npy"), mask)
    
    if exp.complete_mask is not None:
        mask = exp.complete_mask
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask = mask
        np.save(os.path.join(output_dir, f"{base_name.split(".")[0]}_rex_complete_explanation.npy"), mask)  
    
    return mask, exp.sufficiency_confidence




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_folder", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet152")
    parser.add_argument("--script", type=str, default="")
    parser.add_argument("--colour", action="store_true", help="Use the colour of the perturbation as the mask value for ReX")
    parser.add_argument("--no-pred-adv", action="store_true", help="Use the adversarial prediction as the target for ReX")
    args = parser.parse_args()

    print(f"Running ReX on attack folder: {args.attack_folder} with model: {args.model}, script: {args.script}, colour: {args.colour}, no_pred_adv: {args.no_pred_adv}")

    attack_info_files = glob.glob(os.path.join(args.attack_folder, "*.json"))
    if not attack_info_files:
        print(f"No JSON files found in {args.attack_folder}. Skipping.")
        return

    attack_info = attack_info_files[0]
    with open(attack_info, 'r') as f:
        results = json.load(f)

    samples = pd.json_normalize(results.get('samples', []))
    
    OUTPUT_DIR = os.path.join(args.attack_folder, f"{args.model}_{args.colour}_{args.no_pred_adv}_exp")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DUMP_PATH = os.path.join(OUTPUT_DIR, f"{args.model}_attacked_rex.csv")
    # Run on all the Successful attacks
    for idx, sample in tqdm.tqdm(samples.iterrows(), total=samples.shape[0], desc="Processing samples"):
        if sample['attack_success'] is not True:
            continue
        image_name = str(sample['image_name']).replace('.JPEG', '')
        
        
        path_to_image = glob.glob(args.attack_folder + f"/adversarial_examples/*{image_name}*.pt")
        raw_image = torch.load(path_to_image[0])
        print(f"Path to image: {path_to_image[0]},\n {image_name} {sample['attack_success']}, \n {raw_image["metadata"]}")
        # print(f"path_to_image: {raw_image}")
        pred_clean = sample['pred_clean']
        pred_adv = sample['pred_adv']
        image_path = sample['image_path']
        assert image_name == raw_image["metadata"]["image_name"].replace('.JPEG', ''), f"Image name mismatch: {image_name} != {raw_image['metadata']['image_name'].replace('.JPEG', '')}" 
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{image_name}_responsibility.npy")):
            print(f"ReX explanation already exists for {sample['image_name']}, skipping...")
            continue
        if args.colour:
            x, y = np.nonzero(raw_image['perturbation'])[0][1], np.nonzero(raw_image['perturbation'])[0][2]
            print(f"Perturbation location: {x} {y}, {raw_image['adversarial'].shape}")
            print(f"Perturbation values: {raw_image['adversarial'][:, x ,y]}")
            colour = raw_image['adversarial'][:, x ,y]
        else:
            location = None
            colour = None
        print(f"Processing {image_name}: Clean: {pred_clean} -> Adv: {pred_adv} | Colour: {colour}")
        
        output_path = os.path.join(OUTPUT_DIR, f"{image_name}_rex.jpeg")
        if os.path.exists(output_path):
            print(f"ReX output already exists for {image_name}, skipping...")
            continue
        print(f"[{args.model}] Image: {image_name} | Clean: {pred_clean} -> Adv: {pred_adv}")
        addtional_args = []
        if args.colour:
            # create context mask based on the colour of the perturbation
            mask = colour[:, None, None].expand(3, 224, 224).unsqueeze(0)
            mask_path = os.path.join(OUTPUT_DIR, f"{image_name}_mask.npy")
            np.save(mask_path, mask)
            addtional_args = ["--context", mask_path, "--noise", "0.0"]
        if not args.no_pred_adv:
            addtional_args.extend(["--custom_target", str(pred_adv)])
        else:
            addtional_args.extend(["--custom_target", str(pred_clean)])
        try:
            subprocess.run([
                "ReX", image_path,
                "--script", args.script,
                "-v",
                "--dump", DUMP_PATH,
                "--config", "/mnt/data/Documents/ReX/rex.toml",
                "--no_extract",
            ] + addtional_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ReX failed on {image_name}: {e}")
        


if __name__ == "__main__":
    main()
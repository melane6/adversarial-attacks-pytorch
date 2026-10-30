import glob
import os

from captum.attr import LayerGradCam, LayerAttribution, GradientShap, IntegratedGradients, GuidedGradCam
from enum import Enum
import torch
from run_rex_attack import extract_explanation 
import argparse
import json
import pandas as pd
import numpy as np
from PIL import Image
import tqdm
from models import get_gradcam_layer, get_preprocessing, load_model

class XAIMethod(Enum):
    LAYER_GRAD_CAM = "layer_grad_cam"
    GRADIENT_SHAP = "gradient_shap"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GUIDED_GRAD_CAM = "guided_grad_cam"

    def __str__(self):
        return self.value

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def remove_dim(x):
    if x.dim() == 4 and x.shape[0] == 1:
        return x.squeeze(0)
    if x.dim() == 3 and (x.shape[0] == 1 or x.shape[0] == 3):
        return x[0]
    return x

def detach_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError(f"Unsupported type: {type(tensor)}")

def save_attribution(attr, output_dir, name, method):

    np.save(
        os.path.join(output_dir, f"{name}_{method}_raw_attribution.npy"),
        attr,
    )

    norm = normalize(attr)

    np.save(
        os.path.join(output_dir, f"{name}_{method}_normalized_attribution.npy"),
        norm,
    )

    return norm


def load_xai(method_name: XAIMethod, model, **kwargs):
    """
    Load XAI methods
    Args:
        method_name: XAIMethod
        model: Wrapped Pytorch model: Convnext, Resnet, VGG, etc.
        **kwargs:
            layer: name of the layer to use for Grad-CAM and Guided Grad-CAM methods
    Returns:
        Captum XAI method callable that can be used to get the
        attribute of the model (e.g. for the model output)
    """
    if method_name == XAIMethod.LAYER_GRAD_CAM:
        return LayerGradCam(model, kwargs.get("layer"))
    elif method_name == XAIMethod.GRADIENT_SHAP:
        return GradientShap(model)
    elif method_name == XAIMethod.INTEGRATED_GRADIENTS:
        return IntegratedGradients(model)
    elif method_name == XAIMethod.GUIDED_GRAD_CAM:
        return GuidedGradCam(model, layer=kwargs.get("layer"))
    else:
        raise ValueError(f"Unknown XAI method: {method_name}")

def get_xai_method(method_name: str):
    return XAIMethod.value(method_name)

def get_xai_methods():
    return list(XAIMethod)

def explain_model(model, x, y,  method_name: XAIMethod, **kwargs):
    method = load_xai(method_name, model, **kwargs)
    
    if method_name == XAIMethod.LAYER_GRAD_CAM:
        attribution = method.attribute(x, target=y)
        return LayerAttribution.interpolate(attribution, x.shape[-2:])
    elif method_name == XAIMethod.GUIDED_GRAD_CAM:
        return method.attribute(x, target=y, interpolate_mode='bilinear')
    elif method_name == XAIMethod.GRADIENT_SHAP:
        return method.attribute(x, baselines=torch.zeros_like(x), target=y, n_samples=100)
    elif method_name == XAIMethod.INTEGRATED_GRADIENTS:
        attribution, _ = method.attribute(x, target=y, return_convergence_delta=True)
        return attribution

    return method.attribute(x.unsqueeze(0), baselines=None)

def extract_rex(attribution: torch.Tensor, rex_script: str, image_path: str, output_dir: str):
    """
    Extracts the explanation from the attribution tensor using ReX.
    Args:
        attribution: The attribution tensor returned by the XAI method.

    Returns:
        The extracted explanation as a tensor.
    """
    # Normalize the attribution to [0, 1]
    attribution = normalize(attribution)
    
    exp_mask, exp_confidence =  extract_explanation(
        image_path=image_path,  
        output_dir=output_dir,  
        heatmap=attribution,
        rex_script=rex_script 
    )
    print(f"ReX explanation confidence: {exp_confidence}")
    return torch.tensor(exp_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_folder", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet152")
    parser.add_argument("--script", type=str, default="")
    parser.add_argument("--xai_method", type=str, default="layer_grad_cam")
    args = parser.parse_args()

    # set seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, torch.device(device))
    tfm = get_preprocessing(args.model)
    layer = get_gradcam_layer(args.model, model)

    attack_info_files = glob.glob(os.path.join(args.attack_folder, "*.json"))
    if not attack_info_files:
        print(f"No JSON files found in {args.attack_folder}. Skipping.")
        exit(0)

    attack_info = attack_info_files[0]
    with open(attack_info, 'r') as f:
        results = json.load(f)

    samples = pd.json_normalize(results.get('samples', []))
    OUTPUT_DIR = os.path.join(args.attack_folder, f"{args.model}_{args.xai_method}_exp")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, sample in tqdm.tqdm(samples.iterrows(), total=len(samples), desc="Processing samples"):
        image_name = str(sample['image_name']).replace('.JPEG', '')
        class_name = sample['class_name']
        pred_clean = sample['pred_clean']
        pred_adv = sample['pred_adv']
        print(f"Processing {image_name}: class={class_name}, pred_clean={pred_clean}, pred_adv={pred_adv}")
        image_path = sample['image_path']
        output_path = os.path.join(OUTPUT_DIR, f"{image_name}_rex.jpeg")
        if os.path.exists(output_path):
            print(f"ReX output already exists for {image_name}, skipping...")
            continue
        image = Image.open(image_path).convert('RGB')
        image = tfm(image).to(device).unsqueeze(0)
        args.xai_method = XAIMethod(args.xai_method)
        attribution = explain_model(model, image, pred_clean, args.xai_method, layer=layer)
        attribution = remove_dim(attribution)
        attribution = detach_to_numpy(attribution)
        
        # save raw attribution to a numpy file
        attribution = save_attribution(attribution, OUTPUT_DIR, image_name, args.xai_method)

        mask, _ = extract_explanation(
            image_path=image_path,  
            output_dir=OUTPUT_DIR,
            heatmap=attribution,
            rex_script=args.script
        )
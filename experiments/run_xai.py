from captum.attr import LayerGradCam, LayerAttribution, GradientShap, IntegratedGradients, GuidedGradCam, Lime
from enum import Enum
import torch
from run_rex_attack import extract_explanation 

class XAIMethod(Enum):
    LAYER_GRAD_CAM = "layer_grad_cam"
    GRADIENT_SHAP = "gradient_shap"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GUIDED_GRAD_CAM = "guided_grad_cam"

    def __str__(self):
        return self.value


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
    return XAIMethod[method_name]

def get_xai_methods():
    return list(XAIMethod)

def explain_model(model, x, y,  method_name: XAIMethod, **kwargs):
    method = load_xai(method_name, model, **kwargs)
    
    if method_name == XAIMethod.LAYER_GRAD_CAM:
        attribution = method.attribute(x, target=y)
        return LayerAttribution.interpolate(attribution, x.shape[2:])
    elif method_name == XAIMethod.GUIDED_GRAD_CAM:
        return method.attribute(x, target=y, interpolation_mode='bilinear')
    elif method_name == XAIMethod.GRADIENT_SHAP:
        return method.attribute(x, baselines=torch.zeros_like(x), target=y, n_samples=100)
    elif method_name == XAIMethod.INTEGRATED_GRADIENTS:
        attribution, _ = method.attribute(x, target=y, return_convergence_delta=True)
        return attribution

    return method.attribute(x, baselines=None)

def extract_rex(attribution: torch.Tensor, rex_script: str, image_path: str, output_dir: str):
    """
    Extracts the explanation from the attribution tensor using ReX.
    Args:
        attribution: The attribution tensor returned by the XAI method.

    Returns:
        The extracted explanation as a tensor.
    """
    # Normalize the attribution to [0, 1]
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
    
    exp_mask, exp_confidence =  extract_explanation(
        image_path=image_path,  # Replace with actual image path
        output_dir=output_dir,  # Replace with actual output directory
        heatmap=attribution.detach().cpu().numpy(),
        rex_script=rex_script  # Replace with actual ReX script path
    )
    print(f"ReX explanation confidence: {exp_confidence}")
    return torch.tensor(exp_mask)


if __name__ == "__main__":
    # Example usage
    from torchvision.models import resnet18
    import torchvision.transforms as transforms
    from torchattacks import OnePixel
    model = resnet18(pretrained=True)
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    model.eval()

    image_path = "/mnt/data/Documents/ImageNet-Mini/images/n02268443/ILSVRC2012_val_00000418.JPEG"
    x = Image.open(image_path).convert("RGB")
    model = model.to("cuda")
    tfm = []
    tfm.append(transforms.Resize((224, 224)))
    tfm.append(transforms.ToTensor())
    tfm.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tfm = transforms.Compose(tfm)
    x = tfm(x).to("cuda").unsqueeze(0)


    logits_clean = F.softmax(model(x), dim=1)
    preds_score, pred_label = torch.topk(logits_clean, 1)
    print(f"Clean prediction: {pred_label.item()}, Score: {preds_score.item()}")

    attribution = explain_model(model, x, pred_label, XAIMethod.LAYER_GRAD_CAM, layer=model.layer4[-1].conv2)
    # normalize the attribution to [0, 1]
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
    print(f"Attribution shape: {attribution.shape}")
    print(f"Attribution: MAx: {attribution.max()}, Min: {attribution.min()}, Mean: {attribution.mean()}")
    extract_explanation(
        image_path=image_path,  
        output_dir="/mnt/data/Documents/adversarial-attacks-pytorch/temp", 
        heatmap=attribution.detach().cpu().numpy(),
        rex_script="/mnt/data/Documents/ReX/scripts/pytorch_resnet18.py"
    )

    atk = OnePixel(model, pixels=1)
    x_adv = atk(x, pred_label
                ,mask=attribution.detach().cpu().numpy().squeeze(0))
    logits_adv = F.softmax(model(x_adv), dim=1)
    preds_score_adv, pred_label_adv = torch.topk(logits_adv, 1)
    print(f"Clean prediction: {pred_label.item()}, Confidence: {preds_score.item()}")
    print(f"Adversarial prediction: {pred_label_adv.item()}, Confidence: {preds_score_adv.item()}")
    perturbation = (x_adv - x).detach().cpu().numpy()
    print(f"Perturbation: Max: {perturbation.max()}, Min: {perturbation.min()}, Mean: {perturbation.mean()}")
    print(f"Location of the pixel changed: {np.where(perturbation != 0)}")

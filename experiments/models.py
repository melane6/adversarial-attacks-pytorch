from torch import nn
from torchvision import models
import torchvision.transforms as transforms
import torch
# Note: This will download the pretrained models if they are not already available in the cache.
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name: str, device: torch.device, logging=None) -> nn.Module:
    """Load pretrained model."""
    model_map = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'vgg11': models.vgg11,
        'vgg13': models.vgg13,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
        'convnext_large': models.convnext_large,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    if logging is not None:
        logging.info(f"Loading model: {model_name}")
    model = model_map[model_name](weights='DEFAULT')
    model = model.to(device)
    model.eval()

    return model

def get_preprocessing(model_name: str):
    """Get preprocessing transform for model."""
    tfm = []
    if model_name.startswith('resnet') or model_name.startswith('vgg'):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model_name.startswith('convnext'):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    tfm.append(transforms.Resize((224, 224)))
    tfm.append(transforms.ToTensor())
    tfm.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tfm)


import numpy as np
import math

def _validate_and_cast(img1, img2):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    if img1.shape != img2.shape:
        raise ValueError(f"Shapes differ: {img1.shape} vs {img2.shape}")
    # cast to float for safe arithmetic
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # determine dynamic range: prefer [0,1] if max <= 1, else expect [0,255]
    max_val = img1.max()
    if max_val <= 1.0:
        scale = 1.0
    elif max_val <= 255.0:
        scale = 255.0
    else:
        raise ValueError("Image values out of expected ranges [0..1] or [0..255]")
    return img1, img2, scale

def l1_norm(img1, img2):
    img1, img2, scale = _validate_and_cast(img1, img2)
    return float(np.sum(np.abs(img1 - img2))) / (scale * img1.size)

def l2_norm(img1, img2):
    img1, img2, scale = _validate_and_cast(img1, img2)
    return float(np.sqrt(np.sum((img1 - img2) ** 2))) / (scale * math.sqrt(img1.size))

def linf_norm(img1, img2):
    img1, img2, max_value = _validate_and_cast(img1, img2)
    return float(np.max(np.abs(img1 - img2))) / max_value

def l0_norm(img1, img2):
    img1, img2, scale = _validate_and_cast(img1, img2)
    return float(np.sum(img1 != img2)) / img1.size


def compute_norm(img1, img2, norm_type='l2'):
    if norm_type == 'l1':
        return l1_norm(img1, img2)
    elif norm_type == 'l2':
        return l2_norm(img1, img2)
    elif norm_type == 'linf':
        return linf_norm(img1, img2)
    elif norm_type == 'l0':
        return l0_norm(img1, img2)
    else:
        raise ValueError("Unknown norm type")

def compute_ssim(img1, img2):
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, multichannel=True)

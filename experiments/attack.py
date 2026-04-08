from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Callable, Any, Tuple, Optional, TypedDict
import logging
from logging import Logger
import time

import torch
from PIL import Image
from torchvision import models, transforms
from dataset import load_samples, ImageNetRecord
from torch.utils.data import DataLoader, TensorDataset

from torchattacks import OnePixel

MODELS: Dict[str, Callable[..., torch.nn.Module]] = {
    "convnext": models.convnext_large,
    "resnet": models.resnet152,
}

WEIGHTS: Dict[str, Any] = {
    "convnext": models.ConvNeXt_Large_Weights.DEFAULT,
    "resnet": models.ResNet152_Weights.DEFAULT,
}


class AttackSavedDict(TypedDict, total=False):
    adv_inputs: torch.Tensor
    labels: torch.Tensor
    preds: torch.Tensor
    clean_inputs: torch.Tensor
    save_type: str
    # enrichment keys
    image_paths: List[str]
    class_names: List[Optional[str]]
    clean_preds: torch.Tensor
    adv_preds: torch.Tensor
    per_sample_l2: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OnePixel attack on ConvNeXt-Large using samples from dataset."
    )
    p.add_argument(
        "--dataset",
        default="./data/miniimagenet/dataset.json",
        help="Path to dataset.json.",
    )
    p.add_argument(
        "--num-images",
        type=int,
        default=-1,
        help="Number of clean-correct images to test.",
    )
    p.add_argument(
        "--candidate-pool",
        type=int,
        default=2,
        help="Number of candidate images to scan before selecting clean-correct samples.",
    )
    p.add_argument("--pixels", type=int, default=1, help="Number of pixels to perturb.")
    p.add_argument(
        "--steps", type=int, default=20, help="Differential evolution steps."
    )
    p.add_argument("--popsize", type=int, default=20, help="Population size.")
    p.add_argument("--inf-batch", type=int, default=60, help="Inference batch size.")
    p.add_argument(
        "--model",
        type=str,
        default="convnext",
        help="Model to attack.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logging (less verbose than debug).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (very verbose).",
    )
    p.add_argument(
        "--save-path",
        default=None,
        help="Path to save detailed attack results (.pt). Defaults to onepixel_{model}_results.pt",
    )
    return p.parse_args()


def save_and_enrich_results(
    logger: Logger,
    atk: OnePixel,
    model: torch.nn.Module,
    mean: List[float],
    std: List[float],
    selected_images_t: torch.Tensor,
    selected_labels_t: torch.Tensor,
    selected_samples: List[ImageNetRecord],
    selected_clean_pred: torch.Tensor,
    save_path: str,
    device: torch.device,
    inf_batch: int,
    verbose: bool = False,
) -> Tuple[str, AttackSavedDict, float, float]:
    """Run the attack using `Attack.save`, load the saved tensor file, compute model
    predictions on the adversarial examples, compute per-sample L2 distances and
    enrich the saved dict with metadata. Returns (enriched_path, saved_dict, rob_acc, l2).
    """
    save_dataset = TensorDataset(selected_images_t.cpu(), selected_labels_t.cpu())
    save_loader: DataLoader = DataLoader(
        save_dataset, batch_size=inf_batch, shuffle=False
    )

    logger.info("Running OnePixel attack and saving detailed results to %s", save_path)

    t_attack: float = time.time()
    rob_acc_save: float
    l2_save: float
    rob_acc_save, l2_save, _ = atk.save(
        save_loader,
        save_path=save_path,
        verbose=verbose,
        return_verbose=True,
        save_predictions=True,
        save_clean_inputs=True,
        save_type="float",
    )

    logger.info(
        "Attack + save completed in %.2fs (rob_acc=%.2f l2=%.5f)",
        time.time() - t_attack,
        rob_acc_save,
        l2_save,
    )

    saved: AttackSavedDict = torch.load(save_path, map_location="cpu")
    adv_inputs_pixel: torch.Tensor = saved["adv_inputs"]

    # Normalize adversarial inputs for model prediction:
    n_channels: int = len(mean)
    mean_t: torch.Tensor = torch.tensor(mean).reshape(1, n_channels, 1, 1)
    std_t: torch.Tensor = torch.tensor(std).reshape(1, n_channels, 1, 1)
    adv_inputs_norm: torch.Tensor = (adv_inputs_pixel - mean_t) / std_t

    with torch.no_grad():
        pred_adv: torch.Tensor = model(adv_inputs_norm.to(device)).argmax(dim=1).cpu()

    clean_inputs_pixel: torch.Tensor = saved["clean_inputs"]

    per_sample_l2: torch.Tensor = (
        adv_inputs_pixel.view(adv_inputs_pixel.shape[0], -1)
        - clean_inputs_pixel.view(clean_inputs_pixel.shape[0], -1)
    ).norm(p=2, dim=1)

    saved["image_paths"] = [s["image_path"] for s in selected_samples]
    saved["class_names"] = [s.get("class_name") for s in selected_samples]
    saved["clean_preds"] = selected_clean_pred.cpu()
    saved["adv_preds"] = pred_adv
    saved["per_sample_l2"] = per_sample_l2

    enriched_path: str = (
        save_path.replace(".pt", "_enriched.pt")
        if save_path.endswith(".pt")
        else save_path + "_enriched.pt"
    )
    torch.save(saved, enriched_path)

    return enriched_path, saved, rob_acc_save, l2_save


def write_summary(
    logger: Logger,
    saved: AttackSavedDict,
    selected_samples: List[ImageNetRecord],
    selected_labels_t: torch.Tensor,
    selected_clean_pred: torch.Tensor,
    out_path: str = "result.txt",
) -> Tuple[float, List[str]]:
    """Write a human readable summary to `out_path` and return (success_rate, sample_lines)."""
    adv_preds: Optional[torch.Tensor] = saved.get("adv_preds")
    if adv_preds is None:
        adv_preds = saved.get("preds")
    if adv_preds is None:
        raise RuntimeError("No adversarial predictions found in saved results")

    labels_cpu: torch.Tensor = selected_labels_t.cpu()
    success_mask: torch.Tensor = adv_preds.ne(labels_cpu)

    sample_lines: List[str] = []
    for i, sample in enumerate(selected_samples):
        sample_lines.append(
            f"[{i}] {sample['image_path']} | class={sample.get('class_name')} ({sample.get('synset')})"
        )
        sample_lines.append(
            f"    clean_pred={int(selected_clean_pred[i].item())} adv_pred={int(adv_preds[i].item())} "
            f"attack_success={adv_preds[i].item() != labels_cpu[i].item()}"
        )

    success_rate = (
        success_mask.float().mean().item() * 100.0 if len(selected_samples) > 0 else 0.0
    )

    with open(out_path, "w") as f:
        f.write(f"Attack success rate: {success_rate:.2f}%\n")
        for line in sample_lines:
            f.write(line + "\n")

    for line in sample_lines:
        logger.info(line)

    logger.info(
        f"Attack success rate on clean-correct set: {success_rate:.2f}% ({len(selected_samples)} images)."
    )

    return success_rate, sample_lines


def main() -> int:
    args = parse_args()

    dataset_path = Path(args.dataset)
    samples: List[ImageNetRecord] = load_samples(
        dataset_path, max(args.candidate_pool, args.num_images)
    )
    if len(samples) < args.num_images:
        raise RuntimeError(
            f"Requested at least {args.num_images} images but found only {len(samples)} existing files."
        )

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default to WARNING to avoid noisy output; --verbose => INFO, --debug => DEBUG
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("attack")

    logger.info(f"Using device: {device}")

    weights: Any = WEIGHTS[args.model]
    model: torch.nn.Module = MODELS[args.model](weights=weights).to(device).eval()

    # Derive normalisation from the model weights' transforms.
    preprocess = weights.transforms()
    mean: List[float] = list(getattr(preprocess, "mean", [0.485, 0.456, 0.406]))
    std: List[float] = list(getattr(preprocess, "std", [0.229, 0.224, 0.225]))

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    images: List[torch.Tensor] = []
    labels: List[int] = []
    for s in samples:
        img = Image.open(s["image_path"]).convert("RGB")
        images.append(preprocess(img))
        labels.append(int(s["class_id"]))

    images_t: torch.Tensor = torch.stack(images, dim=0).to(device)
    labels_t: torch.Tensor = torch.tensor(labels, dtype=torch.long, device=device)

    atk = OnePixel(
        model,
        pixels=args.pixels,
        steps=args.steps,
        popsize=args.popsize,
        inf_batch=args.inf_batch,
    )
    atk.set_normalization_used(mean=mean, std=std)

    with torch.no_grad():
        logger.info("Computing clean predictions for candidate images...")
        t0: float = time.time()
        clean_logits: torch.Tensor = model(images_t)
        clean_pred: torch.Tensor = clean_logits.argmax(dim=1)
        logger.info(f"Clean predictions computed in {time.time() - t0:.2f}s")

    clean_correct_mask: torch.Tensor = clean_pred.eq(labels_t)
    clean_correct_count: int = int(clean_correct_mask.sum().item())

    # If num_images is -1, use all clean-correct samples.
    if args.num_images == -1:
        args.num_images = clean_correct_count

    if clean_correct_count < args.num_images:
        raise RuntimeError(
            f"Not enough clean-correct candidates. Need {args.num_images}, found {clean_correct_count}."
        )

    selected_indices: torch.Tensor = torch.where(clean_correct_mask)[0][
        : args.num_images
    ]
    selected_samples: List[ImageNetRecord] = [
        samples[i] for i in selected_indices.tolist()
    ]
    selected_images_t: torch.Tensor = images_t[selected_indices]
    selected_labels_t: torch.Tensor = labels_t[selected_indices]
    selected_clean_pred: torch.Tensor = clean_pred[selected_indices]

    logger.info(
        f"Starting OnePixel attack on {len(selected_samples)} images (pixels={args.pixels}, steps={args.steps}, popsize={args.popsize}). This may take a while."
    )

    save_path = (
        args.save_path
        if args.save_path is not None
        else f"onepixel_{args.model}_results.pt"
    )
    enriched_path, saved, rob_acc_save, l2_save = save_and_enrich_results(
        logger,
        atk,
        model,
        mean,
        std,
        selected_images_t,
        selected_labels_t,
        selected_samples,
        selected_clean_pred,
        save_path,
        device,
        args.inf_batch,
        args.verbose,
    )
    logger.info("Detailed attack data saved to %s and %s", save_path, enriched_path)

    write_summary(
        logger,
        saved,
        selected_samples,
        selected_labels_t,
        selected_clean_pred,
        out_path="result.txt",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

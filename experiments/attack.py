from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import logging
import time

import torch
from PIL import Image
from torchvision import models, transforms
from dataset import load_samples

from torchattacks import OnePixel

MODELS = {
    "convnext": models.convnext_large,
    "resnet": models.resnet152,
}

WEIGHTS = {
    "convnext": models.ConvNeXt_Large_Weights.DEFAULT,
    "resnet": models.ResNet152_Weights.DEFAULT,
}


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
        help="Enable debug logging.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = Path(args.dataset)
    samples = load_samples(dataset_path, max(args.candidate_pool, args.num_images))
    if len(samples) < args.num_images:
        raise RuntimeError(
            f"Requested at least {args.num_images} images but found only {len(samples)} existing files."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("attack")
    logger.info(f"Using device: {device}")

    weights = WEIGHTS[args.model]
    model = MODELS[args.model](weights=weights).to(device).eval()

    # Derive normalisation from the model weights' transforms.
    temp_preprocess = weights.transforms()
    mean = list(getattr(temp_preprocess, "mean", [0.485, 0.456, 0.406]))
    std = list(getattr(temp_preprocess, "std", [0.229, 0.224, 0.225]))

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

    images_t = torch.stack(images, dim=0).to(device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

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
        t0 = time.time()
        clean_logits = model(images_t)
        clean_pred = clean_logits.argmax(dim=1)
        logger.info(f"Clean predictions computed in {time.time() - t0:.2f}s")

    clean_correct_mask = clean_pred.eq(labels_t)
    clean_correct_count = int(clean_correct_mask.sum().item())

    # If num_images is -1, use all clean-correct samples.
    if args.num_images == -1:
        args.num_images = clean_correct_count

    if clean_correct_count < args.num_images:
        raise RuntimeError(
            f"Not enough clean-correct candidates. Need {args.num_images}, found {clean_correct_count}."
        )

    selected_indices = torch.where(clean_correct_mask)[0][: args.num_images]
    selected_samples = [samples[i] for i in selected_indices.tolist()]
    selected_images_t = images_t[selected_indices]
    selected_labels_t = labels_t[selected_indices]
    selected_clean_pred = clean_pred[selected_indices]

    logger.info(
        f"Starting OnePixel attack on {len(selected_samples)} images (pixels={args.pixels}, steps={args.steps}, popsize={args.popsize}). This may take a while."
    )
    t_attack = time.time()
    adv = atk(selected_images_t, selected_labels_t)
    logger.info(f"OnePixel attack completed in {time.time() - t_attack:.2f}s")
    with torch.no_grad():
        pred_adv = model(adv).argmax(dim=1)

    best_adv_pred = pred_adv.clone().detach()
    success_mask = pred_adv.ne(selected_labels_t)

    sample_lines: List[str] = []
    for i, sample in enumerate(selected_samples):
        sample_lines.append(
            f"[{i}] {sample['image_path']} | class={sample['class_name']} ({sample['synset']})"
        )
        sample_lines.append(
            f"    clean_pred={selected_clean_pred[i].item()} adv_pred={best_adv_pred[i].item()} "
            f"attack_success={best_adv_pred[i].item() != selected_labels_t[i].item()}"
        )

    success_rate = success_mask.float().mean().item() * 100.0

    logger.info("Results:")
    logger.info(
        f"Selected {len(selected_samples)} clean-correct images from {len(samples)} candidates."
    )
    for line in sample_lines:
        logger.info(line)

    logger.info(
        f"Attack success rate on clean-correct set: {success_rate:.2f}% ({len(selected_samples)} images)."
    )

    with open("result.txt", "w") as f:
        f.write(f"Attack success rate: {success_rate:.2f}%\n")
        for line in sample_lines:
            f.write(line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Run OnePixel attack on ConvNeXt-Large using samples from dataset.json. The attack
is run on a clean-correct subset of the data, and detailed results (including
adversarial examples, predictions, and metadata) are saved to .pt files with batches.
A human-readable summary is also written to result.txt.

The point is to find attackable images on the dataset and identify any patterns
in which samples are vulnerable to a one-pixel attack.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Callable, Any, Tuple, TypedDict
import logging
from logging import Logger
import time

import torch
from PIL import Image
from torchvision import models, transforms
from dataset import load_samples, ImageNetRecord

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
    attack: Dict[str, Any]
    samples: List[Dict[str, Any]]
    robust_accuracy: float
    l2_success_only: float
    batch_index: int
    batch_start: int
    batch_end: int
    batch_files: List[str]


class SummaryRow(TypedDict):
    image_path: str
    class_name: str
    synset: str
    original_classification: int
    adversarial_classification: int
    attack_success: bool


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
        help="Number of images to scan as the candidate pool. Defaults to -1 (scan all samples in dataset.json).",
    )
    p.add_argument(
        "--max-clean-images",
        type=int,
        default=-1,
        help="Maximum number of clean-correct images to use for attack. Defaults to -1 (use all found).",
    )
    p.add_argument("--pixels", type=int, default=1, help="Number of pixels to perturb.")
    p.add_argument(
        "--steps", type=int, default=10, help="Differential evolution steps."
    )
    p.add_argument("--popsize", type=int, default=10, help="Population size.")
    p.add_argument(
        "--attack-batch",
        type=int,
        default=8,
        help="Number of images per attack batch. Each batch is saved to its own .pt file.",
    )
    p.add_argument("--inf-batch", type=int, default=128, help="Inference batch size.")
    p.add_argument(
        "--model",
        type=str,
        default="convnext",
        help="Model to attack.",
    )
    p.add_argument(
        "--save-path",
        default=None,
        help="Path to save detailed attack results (.pt). Defaults to onepixel_{model}_results.pt",
    )
    return p.parse_args()


def configure_logging(log_path: Path) -> Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("attack")
    logger.setLevel(logging.DEBUG)
    return logger


def _batch_save_path(base_save_path: str, batch_index: int) -> str:
    save_path_obj = Path(base_save_path)
    if save_path_obj.suffix:
        return str(
            save_path_obj.with_name(
                f"{save_path_obj.stem}_batch_{batch_index:04d}{save_path_obj.suffix}"
            )
        )
    return str(
        save_path_obj.with_name(f"{save_path_obj.name}_batch_{batch_index:04d}.pt")
    )


def _prepare_data_and_model(
    args: argparse.Namespace,
    logger: Logger,
    device: torch.device,
) -> Tuple[
    torch.nn.Module,
    OnePixel,
    transforms.Compose,
    List[ImageNetRecord],
    torch.Tensor,
]:
    """Load model, preprocess, and select clean-correct samples from dataset.json."""
    dataset_path = Path(args.dataset)
    # If num_images == -1 we load all samples from dataset.json.
    candidate_limit = args.num_images if args.num_images != -1 else -1

    samples: List[ImageNetRecord] = load_samples(dataset_path, candidate_limit)
    scanned_count = len(samples)
    if args.num_images != -1 and scanned_count < args.num_images:
        logger.warning(
            "Requested to scan %d images but dataset contains only %d. Proceeding with %d scanned images.",
            args.num_images,
            scanned_count,
            scanned_count,
        )

    weights: Any = WEIGHTS[args.model]
    model: torch.nn.Module = MODELS[args.model](weights=weights).to(device).eval()

    preprocess = weights.transforms()
    mean: List[float] = list(getattr(preprocess, "mean", [0.485, 0.456, 0.406]))
    std: List[float] = list(getattr(preprocess, "std", [0.229, 0.224, 0.225]))

    logger.info(
        "Model and preprocessing loaded. Model: %s, pixels=%d, steps=%d, popsize=%d, inf_batch=%d",
        args.model,
        args.pixels,
        args.steps,
        args.popsize,
        args.inf_batch,
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    logger.info(
        "Transformed. Model and preprocessing ready. Preparing OnePixel attack instance..."
    )

    atk = OnePixel(
        model,
        pixels=args.pixels,
        steps=args.steps,
        popsize=args.popsize,
        inf_batch=args.inf_batch,
    )
    atk.set_normalization_used(mean=mean, std=std)

    logger.info("Computing clean predictions for candidate images in batches...")
    t0: float = time.time()
    eval_batch = max(1, int(args.inf_batch))
    requested_max_clean = args.max_clean_images
    selected_samples: List[ImageNetRecord] = []
    selected_clean_pred_list: List[int] = []

    with torch.no_grad():
        for start in range(0, len(samples), eval_batch):
            end = min(start + eval_batch, len(samples))
            batch_samples = samples[start:end]

            batch_images: List[torch.Tensor] = []
            batch_labels: List[int] = []
            for sample in batch_samples:
                with Image.open(sample["image_path"]) as img:
                    batch_images.append(preprocess(img.convert("RGB")))
                batch_labels.append(int(sample["class_id"]))

            if not batch_images:
                continue

            images_batch_t = torch.stack(batch_images, dim=0).to(device)
            labels_batch_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
            clean_pred_batch = model(images_batch_t).argmax(dim=1)
            clean_correct_batch = clean_pred_batch.eq(labels_batch_t)

            for local_i in torch.where(clean_correct_batch)[0].tolist():
                selected_samples.append(batch_samples[local_i])
                selected_clean_pred_list.append(int(clean_pred_batch[local_i].item()))

            if (
                requested_max_clean != -1
                and len(selected_samples) >= requested_max_clean
            ):
                selected_samples = selected_samples[:requested_max_clean]
                selected_clean_pred_list = selected_clean_pred_list[
                    :requested_max_clean
                ]
                break

    clean_correct_count = len(selected_samples)
    used_clean_images = clean_correct_count
    logger.info("Clean predictions completed in %.2fs", time.time() - t0)
    logger.info(
        "Scanned %d candidate images, found %d clean-correct; using %d for attack.",
        scanned_count,
        clean_correct_count,
        used_clean_images,
    )

    if clean_correct_count == 0:
        raise RuntimeError(
            "No clean-correct samples found in the scanned candidate pool."
        )

    selected_clean_pred = torch.tensor(selected_clean_pred_list, dtype=torch.long)

    return (
        model,
        atk,
        preprocess,
        selected_samples,
        selected_clean_pred,
    )


def _run_batched_attack_and_save(
    logger: Logger,
    atk: OnePixel,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    selected_samples: List[ImageNetRecord],
    selected_clean_pred: torch.Tensor,
    save_path: str,
    attack_batch: int,
    inf_batch: int,
    attack_metadata: Dict[str, Any],
    device: torch.device,
) -> Tuple[str, AttackSavedDict, float, float, List[SummaryRow]]:
    """Run OnePixel directly and save inline per-sample attack records.

    Returns (save_path, saved_dict, rob_acc, l2).
    """
    logger.info(
        "Running OnePixel attack in batches on %d images (attack_batch=%d)",
        len(selected_samples),
        attack_batch,
    )

    clean_pred_cpu: torch.Tensor = selected_clean_pred.detach().cpu()
    summary_rows: List[SummaryRow] = []
    robust_correct_chunks: List[torch.Tensor] = []
    l2_success_chunks: List[torch.Tensor] = []
    batch_files: List[str] = []

    t_attack_total = time.time()
    for batch_index, start in enumerate(range(0, len(selected_samples), attack_batch)):
        end = min(start + attack_batch, len(selected_samples))
        logger.info(
            "Attacking batch %d: samples [%d:%d)",
            batch_index,
            start,
            end,
        )

        samples_batch = selected_samples[start:end]
        clean_pred_batch_cpu = clean_pred_cpu[start:end]

        batch_images: List[torch.Tensor] = []
        batch_labels: List[int] = []
        for sample in samples_batch:
            with Image.open(sample["image_path"]) as img:
                batch_images.append(preprocess(img.convert("RGB")))
            batch_labels.append(int(sample["class_id"]))

        images_batch = torch.stack(batch_images, dim=0).to(device)
        labels_batch = torch.tensor(batch_labels, dtype=torch.long, device=device)

        t_batch = time.time()
        adv_inputs_norm_batch: torch.Tensor = atk(images_batch, labels_batch)
        logger.info(
            "Batch %d attack finished in %.2fs", batch_index, time.time() - t_batch
        )

        with torch.no_grad():
            pred_adv_parts: List[torch.Tensor] = []
            for pred_start in range(0, adv_inputs_norm_batch.shape[0], inf_batch):
                pred_end = pred_start + inf_batch
                logits = model(adv_inputs_norm_batch[pred_start:pred_end])
                pred_adv_parts.append(logits.argmax(dim=1).detach().cpu())
            pred_adv_batch = (
                torch.cat(pred_adv_parts, dim=0)
                if pred_adv_parts
                else torch.empty((0,), dtype=torch.long)
            )

        labels_batch_cpu = labels_batch.detach().cpu()
        robust_correct_batch = pred_adv_batch.eq(labels_batch_cpu)
        robust_correct_chunks.append(robust_correct_batch)

        delta_batch = (adv_inputs_norm_batch - images_batch).view(
            images_batch.shape[0], -1
        )
        success_mask_batch = ~robust_correct_batch
        batch_success_l2: torch.Tensor = torch.empty((0,), dtype=torch.float)
        if success_mask_batch.any():
            batch_success_l2 = (
                torch.norm(
                    delta_batch[success_mask_batch.to(delta_batch.device)], p=2, dim=1
                )
                .detach()
                .cpu()
            )
            l2_success_chunks.append(batch_success_l2)

        adv_inputs_pixel_batch: torch.Tensor = (
            atk.inverse_normalize(adv_inputs_norm_batch).detach().cpu()
        )

        batch_records: List[Dict[str, Any]] = []
        for local_i, sample in enumerate(samples_batch):
            adv_class = int(pred_adv_batch[local_i].item())
            clean_class = int(clean_pred_batch_cpu[local_i].item())
            batch_records.append(
                {
                    "image_path": sample["image_path"],
                    "original_classification": clean_class,
                    "adversarial_image": adv_inputs_pixel_batch[local_i],
                    "adversarial_classification": adv_class,
                }
            )
            summary_rows.append(
                SummaryRow(
                    image_path=sample["image_path"],
                    class_name=str(sample.get("class_name", "")),
                    synset=str(sample.get("synset", "")),
                    original_classification=clean_class,
                    adversarial_classification=adv_class,
                    attack_success=adv_class != int(labels_batch_cpu[local_i].item()),
                )
            )

        batch_rob_acc = (
            robust_correct_batch.float().mean().item() * 100.0
            if robust_correct_batch.numel() > 0
            else 0.0
        )
        batch_l2 = (
            batch_success_l2.mean().item() if batch_success_l2.numel() > 0 else 0.0
        )

        batch_save_path = _batch_save_path(save_path, batch_index)
        batch_saved: AttackSavedDict = {
            "attack": {**attack_metadata, "attack_batch": attack_batch},
            "batch_index": batch_index,
            "batch_start": start,
            "batch_end": end,
            "samples": batch_records,
            "robust_accuracy": batch_rob_acc,
            "l2_success_only": batch_l2,
        }
        torch.save(batch_saved, batch_save_path)
        batch_files.append(batch_save_path)
        logger.info("Saved batch %d results to %s", batch_index, batch_save_path)

    robust_correct_all = (
        torch.cat(robust_correct_chunks, dim=0)
        if robust_correct_chunks
        else torch.empty((0,), dtype=torch.bool)
    )
    rob_acc_save: float = (
        robust_correct_all.float().mean().item() * 100.0
        if robust_correct_all.numel() > 0
        else 0.0
    )
    l2_save: float = (
        torch.cat(l2_success_chunks, dim=0).mean().item() if l2_success_chunks else 0.0
    )
    logger.info(
        "All batches completed in %.2fs (rob_acc=%.2f l2=%.5f)",
        time.time() - t_attack_total,
        rob_acc_save,
        l2_save,
    )

    saved: AttackSavedDict = {
        "attack": {**attack_metadata, "attack_batch": attack_batch},
        "batch_files": batch_files,
        "robust_accuracy": rob_acc_save,
        "l2_success_only": l2_save,
    }
    torch.save(saved, save_path)
    return save_path, saved, rob_acc_save, l2_save, summary_rows


def write_summary(
    logger: Logger,
    summary_rows: List[SummaryRow],
    out_path: str = "result.txt",
) -> Tuple[float, List[str]]:
    """Write a human readable summary to `out_path` and return (success_rate, sample_lines)."""
    if not summary_rows:
        raise RuntimeError("No sample records found in saved results")

    success_count = sum(1 for row in summary_rows if row["attack_success"])

    sample_lines: List[str] = []
    for i, row in enumerate(summary_rows):
        sample_lines.append(
            f"[{i}] {row['image_path']} | class={row['class_name']} ({row['synset']})"
        )
        sample_lines.append(
            f"    clean_pred={row['original_classification']} adv_pred={row['adversarial_classification']} "
            f"attack_success={row['attack_success']}"
        )

    success_rate = (100.0 * success_count / len(summary_rows)) if summary_rows else 0.0

    with open(out_path, "w") as f:
        f.write(f"Attack success rate: {success_rate:.2f}%\n")
        for line in sample_lines:
            f.write(line + "\n")

    for line in sample_lines:
        logger.info(line)

    logger.info(
        f"Attack success rate on clean-correct set: {success_rate:.2f}% ({len(summary_rows)} images)."
    )

    return success_rate, sample_lines


def main() -> int:
    args = parse_args()

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_save_path = (
        args.save_path
        if args.save_path is not None
        else f"onepixel_{args.model}_results.pt"
    )
    log_path = Path(default_save_path).with_suffix(".log")
    logger = configure_logging(log_path)
    logger.debug("Debug logging is always enabled for both console and file output.")

    logger.info(f"Using device: {device}")

    (
        model,
        atk,
        preprocess,
        selected_samples,
        selected_clean_pred,
    ) = _prepare_data_and_model(args, logger, device)

    logger.info(
        f"Starting OnePixel attack on {len(selected_samples)} images (pixels={args.pixels}, steps={args.steps}, popsize={args.popsize}). This may take a while."
    )

    save_path = (
        args.save_path
        if args.save_path is not None
        else f"onepixel_{args.model}_results.pt"
    )
    saved_path, saved, rob_acc_save, l2_save, summary_rows = (
        _run_batched_attack_and_save(
            logger,
            atk,
            model,
            preprocess,
            selected_samples,
            selected_clean_pred,
            save_path,
            args.attack_batch,
            args.inf_batch,
            attack_metadata={
                "name": "OnePixel",
                "model": args.model,
                "pixels": args.pixels,
                "steps": args.steps,
                "popsize": args.popsize,
                "inf_batch": args.inf_batch,
            },
            device=device,
        )
    )
    logger.info(
        "Saved manifest to %s (rob_acc=%.2f, l2=%.5f). Batch files: %s",
        saved_path,
        rob_acc_save,
        l2_save,
        len(saved.get("batch_files", [])),
    )

    write_summary(
        logger,
        summary_rows,
        out_path="result.txt",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

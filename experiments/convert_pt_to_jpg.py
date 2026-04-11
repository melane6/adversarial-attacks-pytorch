"""
Convert adversarial images from .pt batch files to JPG images.

This script loads .pt files containing OnePixel attack results and extracts
the adversarial images, saving them as individual JPG files.
"""

import argparse
from pathlib import Path
from typing import List
import torch
from PIL import Image
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to a PIL Image.
    
    Assumes tensor is in shape (C, H, W) with values in [0, 1].
    """
    # Ensure tensor is on CPU and detached
    if tensor.device.type != 'cpu':
        tensor = tensor.detach().cpu()
    
    # Clip to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to [0, 255] and change to uint8
    tensor = (tensor * 255).to(torch.uint8)
    
    # Convert from (C, H, W) to (H, W, C)
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
        tensor = tensor.permute(1, 2, 0)
    
    # Handle single channel (grayscale)
    if tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    
    # Convert to numpy and create PIL Image
    numpy_array = tensor.numpy()
    
    if numpy_array.ndim == 2:
        # Grayscale
        img = Image.fromarray(numpy_array, mode='L')
    elif numpy_array.ndim == 3 and numpy_array.shape[-1] == 3:
        # RGB
        img = Image.fromarray(numpy_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    
    return img


def convert_pt_to_jpg(pt_file: Path, output_dir: Path) -> int:
    """Load a .pt file and convert all adversarial images to JPG.
    
    Returns the number of images saved.
    """
    logger.info(f"Loading {pt_file}")
    batch_data = torch.load(pt_file, weights_only=False)
    
    if "samples" not in batch_data:
        logger.warning(f"No 'samples' key in {pt_file}, skipping")
        return 0
    
    samples = batch_data["samples"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_index = batch_data.get("batch_index", 0)
    saved_count = 0
    
    for sample_idx, sample in enumerate(samples):
        if "adversarial_image" not in sample:
            logger.warning(f"Sample {sample_idx} missing 'adversarial_image', skipping")
            continue
        
        adv_image_tensor = sample["adversarial_image"]
        
        try:
            img = tensor_to_image(adv_image_tensor)
            
            # Create output filename from original image path
            orig_path = Path(sample.get("image_path", f"sample_{sample_idx}"))
            output_filename = f"batch_{batch_index:04d}_sample_{sample_idx:04d}_{orig_path.stem}.jpg"
            output_path = output_dir / output_filename
            
            img.save(output_path, quality=95)
            saved_count += 1
            
            if (sample_idx + 1) % 10 == 0:
                logger.info(f"  Saved {sample_idx + 1}/{len(samples)} images from batch {batch_index}")
        
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            continue
    
    logger.info(f"Saved {saved_count} images from {pt_file} to {output_dir}")
    return saved_count


def convert_pt_files(
    input_path: Path,
    output_dir: Path,
) -> int:
    """Convert all .pt files in input_path to JPG.
    
    If input_path is a file, converts that single file.
    If input_path is a directory, converts all .pt files matching the batch pattern.
    
    Returns total number of images saved.
    """
    total_saved = 0
    
    if input_path.is_file():
        if input_path.suffix == '.pt':
            total_saved += convert_pt_to_jpg(input_path, output_dir)
        else:
            logger.error(f"{input_path} is not a .pt file")
    
    elif input_path.is_dir():
        # Look for batch files (e.g., results_batch_0000.pt)
        batch_files = sorted(input_path.glob("*_batch_*.pt"))
        
        if not batch_files:
            # Try direct .pt files
            batch_files = sorted(input_path.glob("*.pt"))
        
        if not batch_files:
            logger.warning(f"No .pt files found in {input_path}")
            return 0
        
        logger.info(f"Found {len(batch_files)} .pt file(s) to convert")
        for pt_file in batch_files:
            total_saved += convert_pt_to_jpg(pt_file, output_dir)
    
    else:
        logger.error(f"{input_path} does not exist")
        return 0
    
    return total_saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert adversarial images from .pt batch files to JPG format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to .pt file or directory containing .pt batch files."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adversarial_images"),
        help="Output directory for JPG files. Defaults to 'adversarial_images'."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    if not args.input.exists():
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    logger.info(f"Converting images from {args.input} to {args.output}")
    total_saved = convert_pt_files(args.input, args.output)
    logger.info(f"Conversion complete. Total images saved: {total_saved}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

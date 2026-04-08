from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, TypedDict, Dict


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG")


class ImageNetRecord(TypedDict):
    image_path: str
    synset: str
    class_name: str
    class_id: int


def get_image_paths(out_dir: str = "./data/miniimagenet") -> List[Path]:
    out = Path(out_dir)
    imgs: List[Path] = []
    for e in IMAGE_EXTS:
        imgs.extend(sorted(out.rglob(e)))
    return imgs


def load_index(out_dir: str = "./data/miniimagenet") -> Dict:
    with open(Path(out_dir) / "ImageNet-Mini/imagenet_class_index.json") as f:
        return json.load(f)


def sample_and_create_dataset_json(
    images: List[Path],
    out_dir: str = "./data/miniimagenet",
    sample_size: int = 500,
    seed: int | None = None,
) -> Path:
    """Randomly sample from all discovered image paths and write simple records.
    Each sample record contains only `image_path` (relative to `out_dir`),
    `synset`, `class_id`, and `class_name`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sample_size = min(sample_size, len(images))
    rng = random.Random(seed)
    sampled_paths = rng.sample(images, sample_size) if sample_size > 0 else []

    index = load_index(out)
    synset_map = {v[0]: (int(k), v[1]) for k, v in index.items()}

    samples = []
    for img in sampled_paths:
        rel = str(img.relative_to(out)).replace("\\", "/")
        synset = img.parent.name
        class_id, class_name = synset_map.get(synset, (-1, ""))
        samples.append(
            {
                "image_path": rel,
                "synset": synset,
                "class_id": class_id,
                "class_name": class_name,
            }
        )

    dataset = {
        "metadata": {
            "version": 3,
            "dataset_root": str(out.resolve()),
            "sample_size": len(samples),
            "seed": seed,
        },
        "samples": samples,
    }

    json_path = out / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)

    return json_path


def get_dataset_json(out_dir: str = "./data/miniimagenet") -> Dict:
    with open(Path(out_dir) / "dataset.json") as f:
        return json.load(f)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Parse mini-ImageNet and create dataset.json."
    )
    p.add_argument("--out", default="./data/miniimagenet", help="Output directory.")
    p.add_argument(
        "--sample-size", type=int, default=500, help="Number of images to include."
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Optional random seed for sampling."
    )
    args = p.parse_args(argv)

    images = get_image_paths(args.out)
    json_path = sample_and_create_dataset_json(
        images, out_dir=args.out, sample_size=args.sample_size, seed=args.seed
    )

    print(
        f"Saved {args.sample_size} samples to {json_path} (found {len(images)} images)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def load_samples(dataset_json_path: Path, max_images: int) -> List[ImageNetRecord]:
    with dataset_json_path.open() as f:
        data = json.load(f)
    samples = data.get("samples", [])
    base_dir = Path(
        data.get("metadata", {}).get("dataset_root", dataset_json_path.parent)
    )

    picked: List[ImageNetRecord] = []
    for s in samples[:max_images]:
        picked.append(
            ImageNetRecord(
                image_path=str(base_dir / s["image_path"]),
                class_id=s.get("class_id"),
                synset=s.get("synset"),
                class_name=s.get("class_name"),
            )
        )
    return picked

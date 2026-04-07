from __future__ import annotations

import argparse
import json
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


def parse_images(images: List[Path], out_dir: str, index: Dict) -> List[ImageNetRecord]:
    out = Path(out_dir)
    synset_map = {v[0]: (int(k), v[1]) for k, v in index.items()}

    records: List[ImageNetRecord] = []
    for img in images:
        synset = img.parent.name
        class_id, class_name = synset_map[synset]
        rel = str(img.relative_to(out)).replace("\\", "/")
        records.append(
            ImageNetRecord(
                image_path=rel,
                synset=synset,
                class_name=class_name,
                class_id=class_id,
            )
        )
    return records


def sample_and_create_dataset_json(
    data: List[ImageNetRecord],
    out_dir: str = "./data/miniimagenet",
    sample_size: int = 500,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sample_size = min(sample_size, len(data))
    sampled = data[:sample_size]

    dataset = {
        "metadata": {
            "version": 2,
            "dataset_root": str(out.resolve()),
            "sample_size": len(sampled),
        },
        "samples": sampled,
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
    args = p.parse_args(argv)

    images = get_image_paths(args.out)
    index = load_index(args.out)
    data = parse_images(images, args.out, index)
    json_path = sample_and_create_dataset_json(
        data, out_dir=args.out, sample_size=args.sample_size
    )

    print(
        f"Saved {args.sample_size} samples to {json_path} (found {len(data)} images)."
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
        img_path = base_dir / s["image_path"]
        picked.append(
            ImageNetRecord(
                image_path=str(img_path),
                class_id=int(s["class_id"]),
                synset=s.get("synset", ""),
                class_name=s.get("class_name", ""),
            )
        )
    return picked

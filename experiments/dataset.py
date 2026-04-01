from __future__ import annotations

from pathlib import Path
from typing import List, TypedDict, Dict


def get_image_paths(out_dir: str = "./data/miniimagenet") -> List[Path]:
    """Get a list of image file paths in `out_dir`."""
    out = Path(out_dir)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG")
    imgs = []
    for e in exts:
        imgs.extend(sorted(out.rglob(e)))
    return imgs


def load_index(out_dir: str = "./data/miniimagenet") -> Dict:
    """Load the ImageNet class index from `out_dir`."""
    import json

    with open(Path(out_dir) / "ImageNet-Mini/imagenet_class_index.json") as f:
        index = json.load(f)
    return index


class ImageNetData(TypedDict):
    image_path: Path
    label: str
    classification: str


def parse_images(images: List[Path], index: Dict):
    """Parse image paths into a list of ImageNetData.

    `index` is of form
    ```
    {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], "2": ["n01484850", "great_white_shark"], ...}
    ```
    """
    data = []
    for img in images:
        label = img.parent.name
        for _, v in index.items():
            if v[0] == label:
                classification = v[1]
                break
        data.append(
            ImageNetData(image_path=img, label=label, classification=classification)
        )
    return data


def sample_and_create_dataset_json(
    data: List[ImageNetData], out_dir: str = "./data/miniimagenet"
):
    """Sample 500 random images from `data` and save to `out_dir/dataset.json`."""
    import json
    import random

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sampled = random.sample(data, 500)
    with open(out / "dataset.json", "w") as f:
        json.dump(sampled, f, default=str, indent=2)


def get_dataset_json(out_dir: str = "./data/miniimagenet") -> List[ImageNetData]:
    """Load the dataset JSON file from `out_dir/dataset.json`.

    Note: To be used by other experiments/scripts.
    """
    import json

    with open(Path(out_dir) / "dataset.json") as f:
        data = json.load(f)
    return [ImageNetData(**d) for d in data]


def main(argv: List[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Parse mini-ImageNet and create dataset.json."
    )

    p.add_argument("--out", default="./data/miniimagenet", help="Output directory.")
    args = p.parse_args(argv)
    images = get_image_paths(args.out)
    index = load_index(args.out)
    data = parse_images(images, index)
    sample_and_create_dataset_json(data, args.out)
    print(
        f"Parsed {len(data)} images and saved dataset.json with 500 samples to {args.out}."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

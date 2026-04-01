from __future__ import annotations

from pathlib import Path
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import List


def download_miniimagenet(
    dataset: str = "deeptrial/miniimagenet", out_dir: str = "./data/miniimagenet"
) -> Path:
    """Download and extract the Kaggle dataset into `out_dir`.

    Requires the `kaggle` package and valid Kaggle credentials.
    """
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Ensure ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY are set"
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files(
        dataset, path=str(out), unzip=True, force=False, quiet=False
    )
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download mini-ImageNet and list images.")
    p.add_argument(
        "--dataset",
        default="deeptrial/miniimagenet",
        help="Kaggle dataset slug (owner/dataset).",
    )
    p.add_argument("--out", default="./data/miniimagenet", help="Output directory.")
    args = p.parse_args(argv)

    out = download_miniimagenet(args.dataset, args.out)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG")
    imgs = []
    for e in exts:
        imgs.extend(sorted(Path(out).rglob(e)))
    print(f"Downloaded/available images under {out}: {len(imgs)}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

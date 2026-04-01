# dataset helpers

- `download_miniimagenet.py` — download a Kaggle mini-ImageNet dataset and list images.
- `dataset.py` — find image files, load the class index, parse image metadata and write a sampled `dataset.json`.
- The `kaggle` package is required only for `download_miniimagenet.py`.
  - Install with: `pip install kaggle`
  - Provide credentials by setting `KAGGLE_API_TOKEN` as environment variable.

1. Download the dataset from Kaggle (optional — if you already have the dataset, skip this):

```bash
pip install kaggle
cd experiments
KAGGLE_API_TOKEN=<token> python download_miniimagenet.py --dataset deeptrial/miniimagenet --out ./data/miniimagenet
```

The script will create `./data/miniimagenet` (if necessary), download and unzip the dataset, and print how many image files were found, or you could just download the dataset manually from Kaggle and place it in `./data/miniimagenet`.

2. Create a sampled dataset JSON file as the "truth"

```bash
cd experiments
python dataset.py --out ./data/miniimagenet
```

This will:

- find image files under `./data/miniimagenet` (searches common image extensions),
- load `ImageNet-Mini/imagenet_class_index.json` from the same directory, and
- sample 500 random entries and write `./data/miniimagenet/dataset.json`.

Output format
The resulting `dataset.json` is a JSON array of objects, see example:

```json
[
	{
		"image_path": "data/miniimagenet/n01440764/n01440764_1.JPEG",
		"label": "n01440764",
		"classification": "tench"
	},
	...
]
```

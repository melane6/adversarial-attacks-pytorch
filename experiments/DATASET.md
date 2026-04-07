# dataset helpers (simplified)

- `download_miniimagenet.py` — optional helper to download the Mini-ImageNet dataset from Kaggle.
- `dataset.py` — produce a small deterministic `dataset.json` that `convnext.py` can load directly.

Install the Kaggle client if you need downloading:

```bash
pip install kaggle
```

Create `dataset.json` (simple, deterministic slice of discovered images):

```bash
cd experiments
python dataset.py --out ./data/miniimagenet --sample-size 500
```

Run ConvNeXt attack using the dataset JSON:

```bash
cd experiments
python convnext.py --dataset-json ./data/miniimagenet/dataset.json --num-images 5
```

Output format (short): `dataset.json` is an object with `metadata` and `samples`. Each sample has `image_path`, `synset`, `class_name`, and `class_id`.

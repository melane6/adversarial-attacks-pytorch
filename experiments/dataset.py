from __future__ import annotations

from pathlib import Path
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path, transform=None, device=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.json_path = dataset_path / "imagenet_class_index.json"
        self.labels = pd.read_json(self.json_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.records_path = None
        self.records = None

        self.load_records()

    def load_records(self):
        self.records_path = self.dataset_path / "dataset.json"
        if self.records_path.exists():
            self.records = pd.read_json(self.records_path)
        else:
            self.create_records()

    def create_records(self):
        img_paths = glob.glob(str(self.dataset_path / "images" / "**/*.JPEG"))
        records = []
        for i, img_path in enumerate(img_paths):
            folder_name = img_path.split("/")[-2]
            print(folder_name)
            idx = np.where(self.labels.values[0] == folder_name)[0][0]
            class_label = self.labels.values[1][idx]
            records.append({
                "image_path": img_path,
                "class_id": idx,
                "class_name": class_label,
                "class_folder": folder_name
            })

        self.records = pd.DataFrame(records)
        self.records.to_json(self.dataset_path / "dataset.json", orient="records")


    def __getitem__(self, index):
        record = self.records.iloc[index]
        image_path = Path(record['image_path'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image.to(self.device),
            'image_path': str(image_path),
            'image_name': image_path.name,
            'image_id': index, # index in dataset.json
            'class_id': record['class_id'],
            'class_name': record['class_name'],
            'class_folder': record['class_folder'],
        }

    def __len__(self):
        return len(self.records)
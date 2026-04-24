from __future__ import annotations

from pathlib import Path
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path, transform=None, device=None, ranking: bool = False,
                 num_exp: int = 1):
        self.dataset_path = dataset_path
        self.transform = transform
        self.json_path = dataset_path / "imagenet_class_index.json"
        self.labels = pd.read_json(self.json_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.records_path = None
        self.records = None
        self.load_records()

        # Explanations - optional
        self.explanations_path = None # ReX csv file path
        self.explanations_folder = None
        self.explanations = None
        self.exp_model = None

        self.ranking = ranking
        self.num_exp = num_exp

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

    def load_explanations(self, path, model_name):
        self.exp_model = model_name
        self.explanations_path = path
        self.explanations_folder = Path(path).parent
        if Path(path).exists():
            self.explanations = pd.read_csv(path)
        else:
            raise ValueError(f"Explanations file not found: {path}")

    def get_exp(self, image_name):
        if self.explanations is None:
            raise ValueError("Explanations not loaded")
        row =  self.explanations[self.explanations['path'].str.contains(image_name)]
        if len(row) == 0:
            return None
        else:
            if self.num_exp > 1:
                return [row[f'explanation_{i}'].values[0] for i in range(self.num_exp)]
            else:
                return row['explanation_0'].values[0] # get the first one out

    def get_ranking(self, image_name):
        if self.explanations is None:
            raise ValueError("Explanations not loaded")
        row =  self.explanations[self.explanations['path'].str.contains(image_name)]
        if len(row) == 0:
            return None
        else:
            return row['responsibility'].values[0] # default to ReX's resp for the time being

    def _exp_shape(self, exp):
        if exp.ndim == 4:
            # (batch, channel, height, width)
            return exp[0]
        else:
            # (channel, height, width)
            return exp

    def process_exp(self, exp):
        if self.num_exp == 1:
            exp = self.explanations_folder.parent / exp
            return torch.from_numpy(np.load(exp)).to(self.device)
        else:
            for i in range(self.num_exp):
                exp[i] = self.explanations_folder.parent / exp[i]
                exp[i] = torch.from_numpy(np.load(exp[i])).to(self.device)
            # combine exps with OR
            ranking = self._exp_shape(exp[0])
            for i in range(len(exp) - 1):
                ranking = ranking | self._exp_shape(exp[i + 1])
            return ranking.to(self.device)


    def __getitem__(self, index):
        record = self.records.iloc[index]
        image_path = Path(record['image_path'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.explanations_path is not None:
            exp = self.get_exp(image_path.name.strip(".JPEG"))
            if exp is not None:
                exp = self.process_exp(exp)
                print(f"Loaded explanation for {image_path.name}: shape {exp.shape}")
        else:
            exp = None
        row =  {
            'image': image.to(self.device),
            'image_path': str(image_path),
            'image_name': image_path.name,
            'image_id': index, # index in dataset.json
            'class_id': record['class_id'],
            'class_name': record['class_name'],
            'class_folder': record['class_folder'],
        }
        if exp is not None:
            row['explanation'] = exp.to(self.device)

        if self.ranking:
            row['ranking'] = self.get_ranking(image_path.name.strip(".JPEG"))
        return row

    def __len__(self):
        return len(self.records)
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
        self.exp_key = "explanation"
        self.exp_key_complete = None

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

    def load_explanations(self, path, model_name, complete_exp=False):
        self.exp_model = model_name
        self.explanations_folder = Path(path)
        if Path(path).is_dir():
            print(f"Loading explanations from {path} for model {model_name}...")
            if complete_exp:
                self.exp_key = "necessity_mask"
                self.exp_key_complete = "complete_mask"
            self.explanations = glob.glob(str(self.explanations_folder / f"*{self.exp_key}*.npy"))
            print(f"Found {self.explanations} explanations")
            if len(self.explanations) == 0:
                raise ValueError(f"No explanations found in {path} with {self.exp_key}")
        else:
            raise ValueError(f"Explanations file not found: {path}")

    def get_exp(self, image_name):
        if self.explanations is None:
            raise ValueError("Explanations not loaded")
        exp_list = [exp for exp in self.explanations if image_name in exp]
        if len(exp_list) == 0:
            print(f"No explanation found for {image_name}")
            return None
        else:
            print(f"Found explanation for {image_name}: {exp_list}")
            if self.num_exp > 1:
                # get _0, _1, _2 etc
                exp_list = [exp for exp in exp_list if f"{self.exp_key}_{self.num_exp-1}" in exp]
                if len(exp_list) == 0:
                    print(f"No explanation found for {image_name} with {self.exp_key}_{self.num_exp-1}")
                    return None
                return exp_list
            else:
                # get _0
                if self.exp_key_complete:
                    return exp_list[0]
                return [exp for exp in exp_list if f"{self.exp_key}_0" in exp][0]

    def get_ranking(self, image_name):
        resp_path = self.get_exp(image_name).replace("explanation_0", "responsibility")
        return resp_path

    def _exp_shape(self, exp):
        if exp.ndim == 4:
            # (batch, channel, height, width)
            return exp.squeeze(0)[0]
        elif exp.ndim == 2:
            # (height, width)
            return exp
        else:
            # (channel, height, width)
            return exp[0]

    def process_exp(self, exp):
        if self.num_exp == 1:
            return self._exp_shape(torch.from_numpy(np.load(exp)).to(self.device))
        else:
            for i in range(self.num_exp):
                exp[i] = self.explanations_folder.parent / exp[i]
                exp[i] = torch.from_numpy(np.load(exp[i])).to(self.device)
            # combine exps with OR
            ranking = self._exp_shape(exp[0])
            for i in range(len(exp) - 1):
                ranking = ranking | self._exp_shape(exp[i + 1])
            return ranking.to(self.device)

    def process_ranking(self, ranking):
        resp = torch.from_numpy(np.load(self.explanations_folder.parent / ranking)).to(self.device)
        print(f"Loaded ranking for {ranking}: shape {resp.shape}")
        return self._exp_shape(resp)


    def __getitem__(self, index):
        record = self.records.iloc[index]
        image_path = Path(record['image_path'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        print(f"Load path: {self.explanations_folder}")
        if self.explanations_folder is not None:
            print(f"Loading explanation for {image_path.name}")
            exp = self.get_exp(image_path.name.strip(".JPEG"))
            print(f"Explanation: {exp}")
            if exp is not None:
                print(f"Processing explanation for {image_path.name}")
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
            ranking_path = self.get_ranking(image_path.name.strip(".JPEG"))
            row['ranking'] = self.process_ranking(ranking_path)
        return row

    def __len__(self):
        return len(self.records)
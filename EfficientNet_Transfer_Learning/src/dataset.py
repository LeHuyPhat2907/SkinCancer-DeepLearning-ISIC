import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SkinDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):

        self.data = pd.read_csv(csv_file)

        # handle CSV dạng 1 cột
        if self.data.shape[1] == 1:
            self.data = self.data.iloc[:, 0].str.split(",", expand=True)
            self.data.columns = ["isic_id", "image_type", "diagnosis_code", "label"]

        self.image_dir = image_dir
        self.transform = transform

        self.data["label"] = self.data["label"].astype(int)

        print(f"[DATASET] Loaded {len(self.data)} samples from {csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 3])

        image_path = os.path.join(self.image_dir, image_name + ".jpg")

        # giảm spam log nhưng vẫn debug được
        if idx % 3000 == 0:
            print(f"[DATASET LOAD] idx={idx} | file={image_name}.jpg")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
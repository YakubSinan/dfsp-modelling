from pathlib import Path
import csv

import cv2
import torch
from torch.utils.data import Dataset


class LocalizationDataset(Dataset):
    def __init__(self, csv_path, image_size=224):
        self.csv_path = Path(csv_path)
        self.image_size = image_size

        with open(self.csv_path, newline="", encoding="utf-8") as file:
            self.records = list(csv.DictReader(file))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = cv2.imread(record["image_path"])

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {record['image_path']}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        image = image.astype("float32") / 255.0

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        )

        bbox = torch.tensor(
            [
                float(record["bbox_center_x_norm"]),
                float(record["bbox_center_y_norm"]),
                float(record["bbox_width_norm"]),
                float(record["bbox_height_norm"]),
            ],
            dtype=torch.float32,
        )

        center = torch.tensor(
            [
                float(record["center_x_norm"]),
                float(record["center_y_norm"]),
            ],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "bbox": bbox,
            "center": center,
        }
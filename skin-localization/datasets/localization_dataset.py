from pathlib import Path
import csv

import cv2
import torch
from torch.utils.data import Dataset


class LocalizationDataset(Dataset):

    def __init__(
        self,
        csv_path,
        image_dir,
        image_size=224,
    ):

        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        with open(
            self.csv_path,
            newline="",
            encoding="utf-8"
        ) as file:

            self.records = list(csv.DictReader(file))

        # Alt klasörler dahil tüm görüntüleri bul
        self.image_paths = {
            image.stem: image
            for image in self.image_dir.rglob("*.jpg")
        }

        print(
            f"Loaded {len(self.image_paths)} images "
            f"from {self.image_dir}"
        )

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_id = record["image_id"]

        image_path = self.image_paths.get(image_id)

        if image_path is None:

            raise FileNotFoundError(
                f"Image ID not found: {image_id}"
            )

        image = cv2.imread(str(image_path))

        if image is None:

            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        image = image.astype("float32") / 255.0

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        )

        # Bounding box:
        # [center_x, center_y, width, height]
        bbox = torch.tensor(
            [
                float(record["bbox_center_x_norm"]),
                float(record["bbox_center_y_norm"]),
                float(record["bbox_width_norm"]),
                float(record["bbox_height_norm"]),
            ],
            dtype=torch.float32,
        )

        # Lesion center
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
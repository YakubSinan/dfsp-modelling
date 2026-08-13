from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DermatofibromaDataset(Dataset):

    def __init__(self, csv_path, image_dir, image_size=224, heatmap_size=7):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)

        self.image_size = image_size
        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = self.image_dir / row["image"]

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Image could not be loaded: {image_path}")

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w = image.shape[:2]

        # Center point
        center_x = float(row["center_x"])
        center_y = float(row["center_y"])

        # Resize image
        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        # Center point'i yeni boyuta taşı
        x = center_x / original_w * self.image_size
        y = center_y / original_h * self.image_size

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        )

        # Heatmap oluştur
        heatmap = np.zeros(
            (self.heatmap_size, self.heatmap_size),
            dtype=np.float32
        )

        # Center point'i heatmap koordinatına taşı
        hx = int(
            x / self.image_size * self.heatmap_size
        )

        hy = int(
            y / self.image_size * self.heatmap_size
        )

        # Sınırlar
        hx = max(0, min(self.heatmap_size - 1, hx))
        hy = max(0, min(self.heatmap_size - 1, hy))

        heatmap[hy, hx] = 1.0

        heatmap = torch.from_numpy(
            heatmap
        ).unsqueeze(0)

        return image, heatmap
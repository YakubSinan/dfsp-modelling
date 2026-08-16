from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DermatofibromaDataset(Dataset):

    def __init__(
        self,
        csv_path,
        image_dir,
        image_size=224,
        heatmap_size=7,
        augment=False
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)

        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = self.image_dir / row["image"]

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Image could not be loaded: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = image.shape[:2]

        center_x = float(row["center_x"])
        center_y = float(row["center_y"])

        # -----------------------------
        # Augmentation
        # -----------------------------

        if self.augment:

            # Horizontal flip
            if np.random.rand() < 0.5:

                image = np.fliplr(image).copy()

                center_x = (
                    original_w - 1
                ) - center_x

            # Vertical flip
            if np.random.rand() < 0.2:

                image = np.flipud(image).copy()

                center_y = (
                    original_h - 1
                ) - center_y

            # Small rotation
            if np.random.rand() < 0.5:

                angle = np.random.uniform(
                    -10,
                    10
                )

                rotation_matrix = cv2.getRotationMatrix2D(
                    (
                        original_w / 2,
                        original_h / 2
                    ),
                    angle,
                    1.0
                )

                image = cv2.warpAffine(
                    image,
                    rotation_matrix,
                    (
                        original_w,
                        original_h
                    ),
                    borderMode=cv2.BORDER_REFLECT_101
                )

                point = np.array(
                    [
                        [center_x, center_y]
                    ],
                    dtype=np.float32
                )

                point = cv2.transform(
                    point[None, :, :],
                    rotation_matrix
                )[0][0]

                center_x = float(point[0])
                center_y = float(point[1])

            # Brightness / contrast
            if np.random.rand() < 0.5:

                alpha = np.random.uniform(
                    0.9,
                    1.1
                )

                beta = np.random.uniform(
                    -10,
                    10
                )

                image = np.clip(
                    image.astype(np.float32) * alpha + beta,
                    0,
                    255
                ).astype(np.uint8)

        # -----------------------------
        # Resize
        # -----------------------------

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size
            )
        )

        # Center point'i yeni boyuta taşı
        x = (
            center_x /
            original_w *
            self.image_size
        )

        y = (
            center_y /
            original_h *
            self.image_size
        )

        # -----------------------------
        # Image tensor
        # -----------------------------

        image = (
            image.astype(np.float32) /
            255.0
        )

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        )

        # -----------------------------
        # Heatmap
        # -----------------------------

        heatmap = np.zeros(
            (
                self.heatmap_size,
                self.heatmap_size
            ),
            dtype=np.float32
        )

        hx = int(
            x /
            self.image_size *
            self.heatmap_size
        )

        hy = int(
            y /
            self.image_size *
            self.heatmap_size
        )

        hx = max(
            0,
            min(
                self.heatmap_size - 1,
                hx
            )
        )

        hy = max(
            0,
            min(
                self.heatmap_size - 1,
                hy
            )
        )

        heatmap[hy, hx] = 1.0

        heatmap = torch.from_numpy(
            heatmap
        ).unsqueeze(0)

        return image, heatmap

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DFSPBBoxDatasetV3(Dataset):

    def __init__(
        self,
        bbox_csv_path,
        image_dir,
        split,
        image_size=224,
        heatmap_size=28,
        augment=False
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.augment = augment

        df = pd.read_csv(bbox_csv_path)

        self.df = (
            df[df["split"] == split]
            .reset_index(drop=True)
        )

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

        # -------------------------------------------------
        # Original bbox
        # -------------------------------------------------

        x_min = float(row["x_min"])
        y_min = float(row["y_min"])
        x_max = float(row["x_max"])
        y_max = float(row["y_max"])

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        # -------------------------------------------------
        # Augmentation
        # -------------------------------------------------

        if self.augment:

            # Horizontal flip
            if np.random.rand() < 0.5:

                image = np.fliplr(image).copy()

                center_x = (
                    original_w - 1
                ) - center_x

                old_x_min = x_min

                x_min = (
                    original_w - 1
                ) - x_max

                x_max = (
                    original_w - 1
                ) - old_x_min

            # Vertical flip
            if np.random.rand() < 0.2:

                image = np.flipud(image).copy()

                center_y = (
                    original_h - 1
                ) - center_y

                old_y_min = y_min

                y_min = (
                    original_h - 1
                ) - y_max

                y_max = (
                    original_h - 1
                ) - old_y_min

            # Rotation
            if np.random.rand() < 0.5:

                angle = np.random.uniform(-10, 10)

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

                corners = np.array(
                    [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ],
                    dtype=np.float32
                )

                transformed = cv2.transform(
                    corners[None, :, :],
                    rotation_matrix
                )[0]

                x_min = float(
                    transformed[:, 0].min()
                )

                x_max = float(
                    transformed[:, 0].max()
                )

                y_min = float(
                    transformed[:, 1].min()
                )

                y_max = float(
                    transformed[:, 1].max()
                )

                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0

            # Brightness / contrast
            if np.random.rand() < 0.5:

                alpha = np.random.uniform(0.9, 1.1)
                beta = np.random.uniform(-10, 10)

                image = np.clip(
                    image.astype(np.float32) * alpha + beta,
                    0,
                    255
                ).astype(np.uint8)

        # -------------------------------------------------
        # Resize image
        # -------------------------------------------------

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size
            )
        )

        # -------------------------------------------------
        # Normalized bbox target
        # -------------------------------------------------

        cx = center_x / original_w
        cy = center_y / original_h

        bw = (x_max - x_min) / original_w
        bh = (y_max - y_min) / original_h

        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        bw = np.clip(bw, 0.0, 1.0)
        bh = np.clip(bh, 0.0, 1.0)

        bbox_target = torch.tensor(
            [cx, cy, bw, bh],
            dtype=torch.float32
        )

        # -------------------------------------------------
        # Image tensor
        # -------------------------------------------------

        image = (
            image.astype(np.float32) / 255.0
        )

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        )

        # -------------------------------------------------
        # 28x28 Gaussian center heatmap
        # -------------------------------------------------

        hx = cx * self.heatmap_size
        hy = cy * self.heatmap_size

        sigma = 1.5

        yy, xx = np.mgrid[
            0:self.heatmap_size,
            0:self.heatmap_size
        ]

        heatmap = np.exp(
            -(
                (xx - hx) ** 2 +
                (yy - hy) ** 2
            ) /
            (2 * sigma ** 2)
        )

        heatmap = torch.from_numpy(
            heatmap.astype(np.float32)
        ).unsqueeze(0)

        return image, heatmap, bbox_target


from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from df_dataset_v3 import DFSPBBoxDatasetV3
from model_v3 import HRNetCenterBBox


ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-5

BBOX_WEIGHT = 1.0

V2_MODEL = ROOT / "dfsp_hrnet_v2_best.pth"
V3_MODEL = ROOT / "dfsp_hrnet_v3_best.pth"


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

train_dataset = DFSPBBoxDatasetV3(
    bbox_csv_path=ROOT / "data" / "dfsp_bbox_annotations.csv",
    image_dir=ROOT / "data" / "dfsp",
    split="train",
    image_size=224,
    heatmap_size=28,
    augment=True
)

val_dataset = DFSPBBoxDatasetV3(
    bbox_csv_path=ROOT / "data" / "dfsp_bbox_annotations.csv",
    image_dir=ROOT / "data" / "dfsp",
    split="val",
    image_size=224,
    heatmap_size=28,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = HRNetCenterBBox().to(DEVICE)


# ---------------------------------------------------------
# Load V2 weights
# ---------------------------------------------------------

if not V2_MODEL.exists():

    raise FileNotFoundError(
        f"V2 model bulunamadı: {V2_MODEL}"
    )

print("Loading V2 checkpoint...")

v2_state = torch.load(
    V2_MODEL,
    map_location=DEVICE
)

model_state = model.state_dict()

loaded = 0
skipped = 0

for key, value in v2_state.items():

    if key in model_state:

        if model_state[key].shape == value.shape:

            model_state[key] = value
            loaded += 1

        else:

            skipped += 1

    else:

        skipped += 1

model.load_state_dict(model_state)

print(
    f"V2 weights loaded: {loaded}"
)

print(
    f"Skipped/new parameters: {skipped}"
)


# ---------------------------------------------------------
# Losses
# ---------------------------------------------------------

center_criterion = nn.BCEWithLogitsLoss()

bbox_criterion = nn.SmoothL1Loss()


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

best_val_loss = float("inf")


print()
print("========================================")
print("DFSP V3 CENTER + BBOX FINE-TUNING")
print("========================================")
print("Device:", DEVICE)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LR)
print("BBox loss weight:", BBOX_WEIGHT)
print("Center heatmap:", "28x28")
print("Augmentation: ON for train")
print("Augmentation: OFF for validation")
print("========================================")
print()


for epoch in range(EPOCHS):

    model.train()

    train_total = 0.0
    train_center = 0.0
    train_bbox = 0.0

    for images, center_targets, bbox_targets in train_loader:

        images = images.to(DEVICE)
        center_targets = center_targets.to(DEVICE)
        bbox_targets = bbox_targets.to(DEVICE)

        optimizer.zero_grad()

        center_pred, bbox_pred = model(images)

        center_loss = center_criterion(
            center_pred,
            center_targets
        )

        bbox_loss = bbox_criterion(
            bbox_pred,
            bbox_targets
        )

        total_loss = (
            center_loss
            +
            BBOX_WEIGHT * bbox_loss
        )

        total_loss.backward()

        optimizer.step()

        train_total += total_loss.item()
        train_center += center_loss.item()
        train_bbox += bbox_loss.item()

    train_total /= len(train_loader)
    train_center /= len(train_loader)
    train_bbox /= len(train_loader)


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    model.eval()

    val_total = 0.0
    val_center = 0.0
    val_bbox = 0.0

    with torch.no_grad():

        for images, center_targets, bbox_targets in val_loader:

            images = images.to(DEVICE)
            center_targets = center_targets.to(DEVICE)
            bbox_targets = bbox_targets.to(DEVICE)

            center_pred, bbox_pred = model(images)

            center_loss = center_criterion(
                center_pred,
                center_targets
            )

            bbox_loss = bbox_criterion(
                bbox_pred,
                bbox_targets
            )

            total_loss = (
                center_loss
                +
                BBOX_WEIGHT * bbox_loss
            )

            val_total += total_loss.item()
            val_center += center_loss.item()
            val_bbox += bbox_loss.item()

    val_total /= len(val_loader)
    val_center /= len(val_loader)
    val_bbox /= len(val_loader)


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Total: {train_total:.4f} "
        f"Center: {train_center:.4f} "
        f"BBox: {train_bbox:.4f} "
        f"| Val Total: {val_total:.4f} "
        f"Center: {val_center:.4f} "
        f"BBox: {val_bbox:.4f}"
    )


    # -----------------------------------------------------
    # Best checkpoint
    # -----------------------------------------------------

    if val_total < best_val_loss:

        best_val_loss = val_total

        torch.save(
            model.state_dict(),
            V3_MODEL
        )

        print(
            "  -> Best V3 model saved."
        )


print()
print("========================================")
print("DFSP V3 TRAINING FINISHED")
print("========================================")
print(
    "Best validation loss:",
    best_val_loss
)
print(
    "Checkpoint:",
    V3_MODEL
)

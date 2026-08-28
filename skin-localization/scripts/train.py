from pathlib import Path
import sys
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# PROJECT IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.localization_dataset import LocalizationDataset
from models.localization_model import LocalizationModel


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10


# ============================================================
# PATHS
# ============================================================

# Project code stays on Google Drive
PROJECT_ROOT = ROOT

# Dataset is read from Colab's fast local storage
LOCAL_DATA_ROOT = Path("/content/skin-localization/data")

TRAIN_CSV = LOCAL_DATA_ROOT / "processed" / "train.csv"
VAL_CSV = LOCAL_DATA_ROOT / "processed" / "val.csv"

IMAGE_DIR = (
    LOCAL_DATA_ROOT
    / "raw"
    / "ham10000"
    / "images"
)

# Results are permanently saved to Google Drive
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "best_localization_model.pth"
METRICS_PATH = OUTPUTS_DIR / "training_metrics.csv"


# ============================================================
# DATASET
# ============================================================

train_dataset = LocalizationDataset(
    TRAIN_CSV,
    IMAGE_DIR,
)

val_dataset = LocalizationDataset(
    VAL_CSV,
    IMAGE_DIR,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


# ============================================================
# MODEL
# ============================================================

model = LocalizationModel(
    pretrained=True
).to(DEVICE)


bbox_loss_fn = nn.SmoothL1Loss()
center_loss_fn = nn.SmoothL1Loss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# METRICS
# ============================================================

def cxcywh_to_xyxy(boxes):
    """
    Converts:
    [center_x, center_y, width, height]

    to:

    [x_min, y_min, x_max, y_max]
    """

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    x_min = cx - width / 2
    y_min = cy - height / 2

    x_max = cx + width / 2
    y_max = cy + height / 2

    return torch.stack(
        [
            x_min,
            y_min,
            x_max,
            y_max,
        ],
        dim=1,
    )


def calculate_iou(pred_bbox, target_bbox):

    pred_boxes = cxcywh_to_xyxy(pred_bbox)
    target_boxes = cxcywh_to_xyxy(target_bbox)

    pred_x1 = pred_boxes[:, 0]
    pred_y1 = pred_boxes[:, 1]
    pred_x2 = pred_boxes[:, 2]
    pred_y2 = pred_boxes[:, 3]

    target_x1 = target_boxes[:, 0]
    target_y1 = target_boxes[:, 1]
    target_x2 = target_boxes[:, 2]
    target_y2 = target_boxes[:, 3]

    inter_x1 = torch.maximum(pred_x1, target_x1)
    inter_y1 = torch.maximum(pred_y1, target_y1)

    inter_x2 = torch.minimum(pred_x2, target_x2)
    inter_y2 = torch.minimum(pred_y2, target_y2)

    inter_width = torch.clamp(
        inter_x2 - inter_x1,
        min=0,
    )

    inter_height = torch.clamp(
        inter_y2 - inter_y1,
        min=0,
    )

    intersection = inter_width * inter_height

    pred_area = (
        torch.clamp(pred_x2 - pred_x1, min=0)
        * torch.clamp(pred_y2 - pred_y1, min=0)
    )

    target_area = (
        torch.clamp(target_x2 - target_x1, min=0)
        * torch.clamp(target_y2 - target_y1, min=0)
    )

    union = pred_area + target_area - intersection

    iou = intersection / (union + 1e-8)

    return iou


def calculate_center_distance(
    predicted_center,
    target_center,
):

    return torch.sqrt(
        torch.sum(
            (predicted_center - target_center) ** 2,
            dim=1,
        )
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_iou = 0.0
    total_center_distance = 0.0
    total_samples = 0

    for batch in train_loader:

        images = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        target_bbox = batch["bbox"].to(
            DEVICE,
            non_blocking=True,
        )

        target_center = batch["center"].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad()

        predicted_bbox, predicted_center = model(images)

        bbox_loss = bbox_loss_fn(
            predicted_bbox,
            target_bbox,
        )

        center_loss = center_loss_fn(
            predicted_center,
            target_center,
        )

        loss = bbox_loss + center_loss

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        iou = calculate_iou(
            predicted_bbox.detach(),
            target_bbox,
        )

        center_distance = calculate_center_distance(
            predicted_center.detach(),
            target_center,
        )

        total_loss += loss.item() * batch_size
        total_iou += iou.sum().item()

        total_center_distance += (
            center_distance.sum().item()
        )

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_iou / total_samples,
        total_center_distance / total_samples,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_center_distance = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(
                DEVICE,
                non_blocking=True,
            )

            target_bbox = batch["bbox"].to(
                DEVICE,
                non_blocking=True,
            )

            target_center = batch["center"].to(
                DEVICE,
                non_blocking=True,
            )

            predicted_bbox, predicted_center = model(images)

            bbox_loss = bbox_loss_fn(
                predicted_bbox,
                target_bbox,
            )

            center_loss = center_loss_fn(
                predicted_center,
                target_center,
            )

            loss = bbox_loss + center_loss

            batch_size = images.size(0)

            iou = calculate_iou(
                predicted_bbox,
                target_bbox,
            )

            center_distance = calculate_center_distance(
                predicted_center,
                target_center,
            )

            total_loss += loss.item() * batch_size
            total_iou += iou.sum().item()

            total_center_distance += (
                center_distance.sum().item()
            )

            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_iou / total_samples,
        total_center_distance / total_samples,
    )


# ============================================================
# TRAINING START
# ============================================================

print("\n========== TRAINING ==========\n")

print(f"Device: {DEVICE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"\nImage directory: {IMAGE_DIR}")


# Create metrics CSV
with open(
    METRICS_PATH,
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "epoch",
            "train_loss",
            "train_iou",
            "train_center_distance",
            "val_loss",
            "val_iou",
            "val_center_distance",
        ]
    )


best_val_loss = float("inf")


for epoch in range(EPOCHS):

    train_loss, train_iou, train_center_distance = (
        train_one_epoch()
    )

    val_loss, val_iou, val_center_distance = (
        validate()
    )

    print(f"\n{'=' * 55}")
    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    print(f"{'=' * 55}")

    print(
        f"Train Loss: {train_loss:.6f} | "
        f"IoU: {train_iou:.4f} | "
        f"Center Error: {train_center_distance:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.6f} | "
        f"IoU: {val_iou:.4f} | "
        f"Center Error: {val_center_distance:.4f}"
    )


    # Save metrics
    with open(
        METRICS_PATH,
        "a",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                epoch + 1,
                train_loss,
                train_iou,
                train_center_distance,
                val_loss,
                val_iou,
                val_center_distance,
            ]
        )


    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_center_distance": val_center_distance,
            },
            BEST_MODEL_PATH,
        )

        print("\n⭐ BEST MODEL SAVED!")

        print(
            f"Validation Loss: "
            f"{val_loss:.6f}"
        )

        print(
            f"Validation IoU: "
            f"{val_iou:.4f}"
        )


print("\n========== TRAINING DONE ==========\n")

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.6f}"
)

print(f"\nMetrics saved to:")
print(METRICS_PATH)

print(f"\nBest model saved to:")
print(BEST_MODEL_PATH)
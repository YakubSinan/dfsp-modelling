from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from datasets.localization_dataset import LocalizationDataset
from models.localization_model import LocalizationModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "best_localization_model.pth"


train_csv = ROOT / "data" / "processed" / "train.csv"
val_csv = ROOT / "data" / "processed" / "val.csv"


train_dataset = LocalizationDataset(train_csv)
val_dataset = LocalizationDataset(val_csv)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


model = LocalizationModel(pretrained=True).to(DEVICE)

bbox_loss_fn = nn.SmoothL1Loss()
center_loss_fn = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


def train_one_epoch():
    model.train()

    total_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(DEVICE)
        target_bbox = batch["bbox"].to(DEVICE)
        target_center = batch["center"].to(DEVICE)

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

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate():
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            target_bbox = batch["bbox"].to(DEVICE)
            target_center = batch["center"].to(DEVICE)

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

            total_loss += loss.item()

    return total_loss / len(val_loader)


print("\n========== TRAINING ==========\n")

print(f"Device: {DEVICE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    train_loss = train_one_epoch()

    val_loss = validate()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            BEST_MODEL_PATH,
        )

        print(f"Best model saved: {BEST_MODEL_PATH}")


print("\n========== TRAINING DONE ==========\n")

print(f"Best validation loss: {best_val_loss:.6f}")
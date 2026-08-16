from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

from df_dataset import DermatofibromaDataset


ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-5


# -----------------------------
# Dataset
# -----------------------------

train_dataset = DermatofibromaDataset(
    csv_path=ROOT / "data" / "dfsp_train.csv",
    image_dir=ROOT / "data" / "dfsp",
    augment=True
)

val_dataset = DermatofibromaDataset(
    csv_path=ROOT / "data" / "dfsp_val.csv",
    image_dir=ROOT / "data" / "dfsp",
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


# -----------------------------
# HRNet
# -----------------------------

class HRNetLocalization(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=False,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Conv2d(
                feature_dim,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                256,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):

        features = self.backbone.forward_features(x)

        x = self.head(features)

        return x


model = HRNetLocalization().to(DEVICE)


# -----------------------------
# Load DF pretrained model
# -----------------------------

PRETRAINED_MODEL = ROOT / "df_hrnet_best.pth"

if not PRETRAINED_MODEL.exists():
    raise FileNotFoundError(
        f"DF pretrained model bulunamadı: {PRETRAINED_MODEL}"
    )

print("Loading DF pretrained model...")

state_dict = torch.load(
    PRETRAINED_MODEL,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

print("DF pretrained model loaded.")


# -----------------------------
# Loss + optimizer
# -----------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# -----------------------------
# Training
# -----------------------------

best_val_loss = float("inf")

print()
print("================================")
print("DFSP FINE-TUNING")
print("================================")
print("Device:", DEVICE)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LR)
print("Augmentation: ON for train")
print("Augmentation: OFF for validation")
print("================================")
print()


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for images, targets in train_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)

        targets = F.interpolate(
            targets,
            size=predictions.shape[-2:],
            mode="nearest"
        )

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(images)

            targets = F.interpolate(
                targets,
                size=predictions.shape[-2:],
                mode="nearest"
            )

            loss = criterion(
                predictions,
                targets
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # -------------------------
    # Best checkpoint
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            ROOT / "dfsp_hrnet_best.pth"
        )

        print(
            "  -> Best DFSP model saved."
        )


print()
print("DFSP fine-tuning finished.")
print("Best validation loss:", best_val_loss)
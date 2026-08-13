from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from df_dataset import DermatofibromaDataset


ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4


# -----------------------------
# Dataset
# -----------------------------

train_dataset = DermatofibromaDataset(
    csv_path=ROOT / "data" / "df_train.csv",
    image_dir=ROOT / "data" / "dermatofibroma"
)

val_dataset = DermatofibromaDataset(
    csv_path=ROOT / "data" / "df_val.csv",
    image_dir=ROOT / "data" / "dermatofibroma"
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
            pretrained=True,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):

        features = self.backbone.forward_features(x)

        # HRNet feature map
        x = self.head(features)

        return x


model = HRNetLocalization().to(DEVICE)


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

print("Device:", DEVICE)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for images, targets in train_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)

        # Targetı model çıktısına resize et
        targets = torch.nn.functional.interpolate(
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

            targets = torch.nn.functional.interpolate(
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
            ROOT / "df_hrnet_best.pth"
        )

        print("  -> Best model saved.")


print("\nTraining finished.")
print("Best validation loss:", best_val_loss)
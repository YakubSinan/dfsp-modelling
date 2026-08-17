
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm


ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CSV = ROOT / "data" / "dfsp_test.csv"
IMAGE_DIR = ROOT / "data" / "dfsp"
MODEL_PATH = ROOT / "dfsp_hrnet_best.pth"


# -----------------------------
# Model
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


# -----------------------------
# Load model
# -----------------------------

print("Loading DFSP model...")

model = HRNetLocalization().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()

print("DFSP model loaded.")
print("Device:", DEVICE)


# -----------------------------
# Load test data
# -----------------------------

df = pd.read_csv(TEST_CSV)

print("Test samples:", len(df))


# -----------------------------
# Evaluation
# -----------------------------

errors = []

results = []


for _, row in df.iterrows():

    image_name = row["image"]

    image_path = IMAGE_DIR / image_name

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Image could not be loaded: {image_path}"
        )

    original_height, original_width = image.shape[:2]


    # -------------------------
    # Preprocess
    # -------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        image_rgb,
        (224, 224)
    )

    tensor = torch.from_numpy(
        resized
    ).float() / 255.0

    tensor = tensor.permute(
        2, 0, 1
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)


    # -------------------------
    # Prediction
    # -------------------------

    with torch.no_grad():

        prediction = model(tensor)

        heatmap = torch.sigmoid(
            prediction
        )[0, 0].cpu().numpy()


    # -------------------------
    # Predicted center
    # -------------------------

    pred_y, pred_x = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape
    )


    # Convert 7x7 heatmap coordinates
    # back to original image coordinates

    pred_x = (
        pred_x / (heatmap.shape[1] - 1)
    ) * (original_width - 1)

    pred_y = (
        pred_y / (heatmap.shape[0] - 1)
    ) * (original_height - 1)


    # -------------------------
    # Ground truth
    # -------------------------

    gt_x = float(row["center_x"])
    gt_y = float(row["center_y"])


    # -------------------------
    # Euclidean error
    # -------------------------

    error = np.sqrt(
        (pred_x - gt_x) ** 2 +
        (pred_y - gt_y) ** 2
    )

    errors.append(error)


    results.append({
        "image": image_name,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "pred_x": round(pred_x, 2),
        "pred_y": round(pred_y, 2),
        "error_px": round(error, 2)
    })


# -----------------------------
# Metrics
# -----------------------------

errors = np.array(errors)

print()
print("================================")
print("DFSP TEST RESULTS")
print("================================")

print(
    f"Mean localization error: "
    f"{errors.mean():.2f} px"
)

print(
    f"Median localization error: "
    f"{np.median(errors):.2f} px"
)

print(
    f"Max localization error: "
    f"{errors.max():.2f} px"
)

print(
    f"Accuracy <= 10 px: "
    f"{(errors <= 10).mean() * 100:.1f}%"
)

print(
    f"Accuracy <= 20 px: "
    f"{(errors <= 20).mean() * 100:.1f}%"
)

print(
    f"Accuracy <= 30 px: "
    f"{(errors <= 30).mean() * 100:.1f}%"
)

print()
print("Per-image results:")
print()

for result in results:

    print(
        f"{result['image']} | "
        f"GT=({result['gt_x']:.0f}, {result['gt_y']:.0f}) | "
        f"Pred=({result['pred_x']:.0f}, {result['pred_y']:.0f}) | "
        f"Error={result['error_px']:.2f}px"
    )


# -----------------------------
# Save results
# -----------------------------

results_df = pd.DataFrame(results)

output_path = ROOT / "dfsp_test_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print()
print("Results saved to:")
print(output_path)

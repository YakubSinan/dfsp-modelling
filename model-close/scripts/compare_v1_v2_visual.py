from pathlib import Path
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = ROOT / "data" / "dfsp"
BBOX_CSV = ROOT / "data" / "dfsp_bbox_annotations.csv"

V1_MODEL = ROOT / "dfsp_hrnet_best.pth"
V2_MODEL = ROOT / "dfsp_hrnet_v2_best.pth"

OUTPUT_DIR = ROOT / "v1_v2_visual_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class HRNetLocalization(nn.Module):

    def __init__(self, output_size):

        super().__init__()

        self.output_size = output_size

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

        x = F.interpolate(
            x,
            size=(
                self.output_size,
                self.output_size
            ),
            mode="bilinear",
            align_corners=False
        )

        return x


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

def load_model(path):

    model = HRNetLocalization(
        output_size=28
    ).to(DEVICE)

    state = torch.load(
        path,
        map_location=DEVICE
    )

    model.load_state_dict(state)

    model.eval()

    return model


print("Device:", DEVICE)

print("Loading V1...")
v1 = load_model(V1_MODEL)

print("Loading V2...")
v2 = load_model(V2_MODEL)

print("Models loaded.")


# ---------------------------------------------------------
# Test CSV
# ---------------------------------------------------------

rows = []

with open(
    BBOX_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        if row["split"] == "test":
            rows.append(row)


print()
print("Test samples:", len(rows))


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def preprocess(image):

    original_h, original_w = image.shape[:2]

    resized = cv2.resize(
        image,
        (224, 224)
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        rgb.astype(np.float32) / 255.0
    )

    tensor = torch.from_numpy(
        tensor.transpose(2, 0, 1)
    )

    tensor = tensor.unsqueeze(0).to(
        DEVICE
    )

    return tensor, original_w, original_h


# ---------------------------------------------------------
# Soft argmax
# ---------------------------------------------------------

def soft_argmax(heatmap):

    heatmap = heatmap.squeeze()

    heatmap = torch.sigmoid(
        heatmap
    )

    h, w = heatmap.shape

    flat = heatmap.reshape(-1)

    weights = flat / (
        flat.sum() + 1e-8
    )

    yy, xx = torch.meshgrid(
        torch.arange(
            h,
            device=heatmap.device
        ),
        torch.arange(
            w,
            device=heatmap.device
        ),
        indexing="ij"
    )

    x = (
        (weights * xx.reshape(-1))
        .sum()
    )

    y = (
        (weights * yy.reshape(-1))
        .sum()
    )

    return (
        x.item(),
        y.item()
    )


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

results = []

for i, row in enumerate(rows):

    name = row["image"]

    image_path = IMAGE_DIR / name

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "❌ Could not read:",
            name
        )

        continue

    input_tensor, original_w, original_h = preprocess(
        image
    )

    with torch.no_grad():

        pred_v1 = v1(
            input_tensor
        )

        pred_v2 = v2(
            input_tensor
        )

    x1, y1 = soft_argmax(
        pred_v1
    )

    x2, y2 = soft_argmax(
        pred_v2
    )

    # 28x28 → 224x224
    x1 *= 224 / 28
    y1 *= 224 / 28

    x2 *= 224 / 28
    y2 *= 224 / 28

    # 224x224 → original image
    x1 *= original_w / 224
    y1 *= original_h / 224

    x2 *= original_w / 224
    y2 *= original_h / 224

    # Ground truth bbox
    bx1 = float(row["x_min"])
    by1 = float(row["y_min"])
    bx2 = float(row["x_max"])
    by2 = float(row["y_max"])

    gt_cx = (
        bx1 + bx2
    ) / 2

    gt_cy = (
        by1 + by2
    ) / 2

    v1_inside = (
        bx1 <= x1 <= bx2
        and
        by1 <= y1 <= by2
    )

    v2_inside = (
        bx1 <= x2 <= bx2
        and
        by1 <= y2 <= by2
    )

    # -----------------------------------------------------
    # Draw
    # -----------------------------------------------------

    canvas = image.copy()

    # Ground truth bbox
    cv2.rectangle(
        canvas,
        (int(bx1), int(by1)),
        (int(bx2), int(by2)),
        (0, 255, 0),
        2
    )

    # Ground truth center
    cv2.circle(
        canvas,
        (int(gt_cx), int(gt_cy)),
        5,
        (0, 255, 0),
        -1
    )

    # V1 = blue
    cv2.circle(
        canvas,
        (int(x1), int(y1)),
        7,
        (255, 0, 0),
        -1
    )

    # V2 = red
    cv2.circle(
        canvas,
        (int(x2), int(y2)),
        7,
        (0, 0, 255),
        -1
    )

    # Labels
    cv2.putText(
        canvas,
        f"V1: ({x1:.1f}, {y1:.1f}) "
        f"{'INSIDE' if v1_inside else 'OUTSIDE'}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        f"V2: ({x2:.1f}, {y2:.1f}) "
        f"{'INSIDE' if v2_inside else 'OUTSIDE'}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    output_path = (
        OUTPUT_DIR /
        f"{i:02d}_{name}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        canvas
    )

    results.append({
        "image": name,
        "v1_x": x1,
        "v1_y": y1,
        "v2_x": x2,
        "v2_y": y2,
        "bbox_x_min": bx1,
        "bbox_y_min": by1,
        "bbox_x_max": bx2,
        "bbox_y_max": by2,
        "v1_inside_bbox": v1_inside,
        "v2_inside_bbox": v2_inside
    })

    print(
        f"{name}"
    )

    print(
        f"  V1: "
        f"({x1:.1f}, {y1:.1f}) "
        f"| inside={v1_inside}"
    )

    print(
        f"  V2: "
        f"({x2:.1f}, {y2:.1f}) "
        f"| inside={v2_inside}"
    )


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

output_csv = (
    ROOT /
    "v1_v2_visual_results.csv"
)

with open(
    output_csv,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()

    writer.writerows(results)


v1_hits = sum(
    r["v1_inside_bbox"]
    for r in results
)

v2_hits = sum(
    r["v2_inside_bbox"]
    for r in results
)


print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"V1 bbox hit-rate: "
    f"{v1_hits}/{len(results)} "
    f"= {100*v1_hits/len(results):.1f}%"
)

print(
    f"V2 bbox hit-rate: "
    f"{v2_hits}/{len(results)} "
    f"= {100*v2_hits/len(results):.1f}%"
)

print()
print(
    "Visual results:",
    OUTPUT_DIR
)

print(
    "CSV:",
    output_csv
)
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm


IMAGE_PATH = Path("data/dermatofibroma/ISIC_0024553.jpg")
OUTPUT_PATH = Path("smoke_test_result.jpg")


class HRNetLocalization(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            features_only=True,
        )

        # HRNet-W18 final feature map: 512 channels
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, x):
        features = self.backbone(x)
        feature_map = features[-1]

        heatmap = self.heatmap_head(feature_map)
        heatmap = torch.sigmoid(heatmap)

        return heatmap


# --------------------------------------------------
# 1. Image
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

original = image.copy()

print("Image loaded:", IMAGE_PATH)
print("Image shape:", image.shape)


# --------------------------------------------------
# 2. Model
# --------------------------------------------------

device = torch.device("cpu")

model = HRNetLocalization()
model.to(device)
model.eval()

print("HRNet localization model loaded!")


# --------------------------------------------------
# 3. Preprocess
# --------------------------------------------------

resized = cv2.resize(image, (224, 224))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

tensor = torch.from_numpy(rgb).float()
tensor = tensor.permute(2, 0, 1)
tensor = tensor.unsqueeze(0) / 255.0
tensor = tensor.to(device)


# --------------------------------------------------
# 4. Inference
# --------------------------------------------------

with torch.no_grad():
    heatmap = model(tensor)

print("Heatmap generated!")
print("Heatmap shape:", heatmap.shape)


# --------------------------------------------------
# 5. Find center point
# --------------------------------------------------

heatmap_np = heatmap[0, 0].cpu().numpy()

y, x = np.unravel_index(
    np.argmax(heatmap_np),
    heatmap_np.shape
)

# Convert 224x224 coordinates back to original image
original_h, original_w = original.shape[:2]

center_x = int(x * original_w / 224)
center_y = int(y * original_h / 224)

print(f"Predicted center: ({center_x}, {center_y})")


# --------------------------------------------------
# 6. Draw predicted point
# --------------------------------------------------

cv2.circle(
    original,
    (center_x, center_y),
    10,
    (0, 255, 0),
    -1
)

cv2.putText(
    original,
    f"Center: ({center_x}, {center_y})",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)


# --------------------------------------------------
# 7. Save result
# --------------------------------------------------

cv2.imwrite(str(OUTPUT_PATH), original)

print("Result saved:", OUTPUT_PATH)
print("SMOKE TEST PASSED")
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from datasets.localization_dataset import LocalizationDataset
from models.localization_model import LocalizationModel


csv_path = ROOT / "data" / "processed" / "train.csv"

dataset = LocalizationDataset(csv_path)

sample = dataset[0]

image = sample["image"].unsqueeze(0)

model = LocalizationModel(pretrained=False)

model.eval()

with torch.no_grad():
    bbox, center = model(image)

print("\n========== MODEL TEST ==========\n")

print(f"Input shape: {image.shape}")
print(f"BBox prediction: {bbox}")
print(f"Center prediction: {center}")

print("\n========== DONE ==========\n")
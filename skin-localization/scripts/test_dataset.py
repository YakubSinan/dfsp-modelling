from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from datasets.localization_dataset import LocalizationDataset


csv_path = ROOT / "data" / "processed" / "train.csv"

dataset = LocalizationDataset(csv_path)

print("\n========== DATASET TEST ==========\n")

print(f"Total samples: {len(dataset)}")

sample = dataset[0]

print(f"\nImage shape: {sample['image'].shape}")
print(f"BBox: {sample['bbox']}")
print(f"Center: {sample['center']}")

print("\n========== DONE ==========\n")
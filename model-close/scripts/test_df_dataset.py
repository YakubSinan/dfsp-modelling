from pathlib import Path

from df_dataset import DermatofibromaDataset


ROOT = Path(__file__).resolve().parents[1]

dataset = DermatofibromaDataset(
    csv_path=ROOT / "data" / "df_train.csv",
    image_dir=ROOT / "data" / "dermatofibroma"
)

print("Dataset size:", len(dataset))

image, heatmap = dataset[0]

print("Image shape:", image.shape)
print("Heatmap shape:", heatmap.shape)
print("Heatmap max:", heatmap.max().item())
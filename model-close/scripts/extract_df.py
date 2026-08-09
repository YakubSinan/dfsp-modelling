import os
import shutil
import pandas as pd

# Proje kök dizini
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# HAM10000 yolları
metadata_path = os.path.join(ROOT, "data", "raw", "ham10000", "HAM10000_metadata.csv")

image_dirs = [
    os.path.join(ROOT, "data", "raw", "ham10000", "HAM10000_images_part_1"),
    os.path.join(ROOT, "data", "raw", "ham10000", "HAM10000_images_part_2"),
]

# Çıkış klasörü
output_dir = os.path.join(ROOT, "model-close", "data", "dermatofibroma")
os.makedirs(output_dir, exist_ok=True)

# CSV oku
df = pd.read_csv(metadata_path)

# Sadece Dermatofibroma
df = df[df["dx"] == "df"]

print(f"Found {len(df)} Dermatofibroma images.")

copied = 0

for image_id in df["image_id"]:
    filename = image_id + ".jpg"

    for folder in image_dirs:
        src = os.path.join(folder, filename)

        if os.path.exists(src):
            shutil.copy(src, output_dir)
            copied += 1
            break

print(f"Copied {copied} images.")
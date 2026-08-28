from pathlib import Path
import csv

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = ROOT / "data" / "raw" / "ham10000" / "images"
MASKS_DIR = ROOT / "data" / "raw" / "ham10000_masks"

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = {
    image.stem: image
    for image in IMAGES_DIR.rglob("*.jpg")
}

masks = {
    mask.stem.replace("_segmentation", ""): mask
    for mask in MASKS_DIR.rglob("*.png")
}

matched_ids = sorted(images.keys() & masks.keys())

output_csv = OUTPUT_DIR / "ham10000_localization.csv"

records = []

for index, image_id in enumerate(matched_ids, start=1):
    image_path = images[image_id]
    mask_path = masks[image_id]

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    height, width = image.shape[:2]

    binary_mask = (mask > 0).astype(np.uint8)

    y_coords, x_coords = np.where(binary_mask > 0)

    if len(x_coords) == 0 or len(y_coords) == 0:
        continue

    x_min = int(x_coords.min())
    y_min = int(y_coords.min())
    x_max = int(x_coords.max())
    y_max = int(y_coords.max())

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    bbox_center_x = (x_min + x_max) / 2
    bbox_center_y = (y_min + y_max) / 2

    moments = cv2.moments(binary_mask)

    if moments["m00"] == 0:
        continue

    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]

    records.append({
        "image_id": image_id,
        "image_path": str(image_path),
        "mask_path": str(mask_path),

        "image_width": width,
        "image_height": height,

        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,

        "bbox_width": bbox_width,
        "bbox_height": bbox_height,

        "bbox_center_x": round(bbox_center_x, 4),
        "bbox_center_y": round(bbox_center_y, 4),

        "bbox_center_x_norm": round(bbox_center_x / width, 6),
        "bbox_center_y_norm": round(bbox_center_y / height, 6),

        "center_x": round(center_x, 4),
        "center_y": round(center_y, 4),

        "center_x_norm": round(center_x / width, 6),
        "center_y_norm": round(center_y / height, 6),

        "bbox_width_norm": round(bbox_width / width, 6),
        "bbox_height_norm": round(bbox_height / height, 6),
    })

    if index % 500 == 0:
        print(f"Processed {index}/{len(matched_ids)}")

fieldnames = [
    "image_id",
    "image_path",
    "mask_path",

    "image_width",
    "image_height",

    "x_min",
    "y_min",
    "x_max",
    "y_max",

    "bbox_width",
    "bbox_height",

    "bbox_center_x",
    "bbox_center_y",
    "bbox_center_x_norm",
    "bbox_center_y_norm",

    "center_x",
    "center_y",
    "center_x_norm",
    "center_y_norm",

    "bbox_width_norm",
    "bbox_height_norm",
]

with open(output_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"\nProcessed samples: {len(records)}")
print(f"Saved: {output_csv}")
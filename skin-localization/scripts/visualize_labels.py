from pathlib import Path
import csv
import random

import cv2

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "data" / "processed" / "ham10000_localization.csv"
OUTPUT_DIR = ROOT / "outputs" / "label_visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8") as file:
    records = list(csv.DictReader(file))

samples = random.sample(records, min(10, len(records)))

for record in samples:
    image = cv2.imread(record["image_path"])

    if image is None:
        continue

    x_min = int(record["x_min"])
    y_min = int(record["y_min"])
    x_max = int(record["x_max"])
    y_max = int(record["y_max"])

    center_x = int(float(record["center_x"]))
    center_y = int(float(record["center_y"]))

    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2
    )

    cv2.circle(
        image,
        (center_x, center_y),
        6,
        (0, 0, 255),
        -1
    )

    output_path = OUTPUT_DIR / f"{record['image_id']}_visualization.jpg"

    cv2.imwrite(str(output_path), image)

print(f"Created {len(samples)} visualizations")
print(f"Saved to: {OUTPUT_DIR}")
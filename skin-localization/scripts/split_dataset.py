from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "data" / "processed" / "ham10000_localization.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

SEED = 42

with open(INPUT_CSV, newline="", encoding="utf-8") as file:
    records = list(csv.DictReader(file))

random.seed(SEED)
random.shuffle(records)

total = len(records)

train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_records = records[:train_end]
val_records = records[train_end:val_end]
test_records = records[val_end:]

splits = {
    "train": train_records,
    "val": val_records,
    "test": test_records,
}

fieldnames = records[0].keys()

for split_name, split_records in splits.items():
    output_path = OUTPUT_DIR / f"{split_name}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_records)

print("\n========== DATASET SPLIT ==========\n")

for split_name, split_records in splits.items():
    print(f"{split_name.capitalize()}: {len(split_records)}")

print(f"\nTotal: {total}")
print("\n========== DONE ==========\n")
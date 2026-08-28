from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = ROOT / "data" / "raw" / "ham10000" / "images"
MASKS_DIR = ROOT / "data" / "raw" / "ham10000_masks"

images = list(IMAGES_DIR.rglob("*.jpg"))
masks = list(MASKS_DIR.rglob("*.png"))

print("\n========== DATASET CHECK ==========\n")

print(f"Images directory: {IMAGES_DIR}")
print(f"Masks directory: {MASKS_DIR}\n")

print(f"Total images: {len(images)}")
print(f"Total masks: {len(masks)}")

image_ids = {img.stem for img in images}

mask_ids = {
    mask.stem.replace("_segmentation", "")
    for mask in masks
}

matched = image_ids & mask_ids

missing_masks = image_ids - mask_ids
missing_images = mask_ids - image_ids

print("\n========== MATCHING ==========\n")

print(f"Matched image-mask pairs: {len(matched)}")
print(f"Images without masks: {len(missing_masks)}")
print(f"Masks without images: {len(missing_images)}")

if images:
    print("\n========== IMAGE SAMPLE ==========\n")
    print(images[0])

if masks:
    print("\n========== MASK SAMPLE ==========\n")
    print(masks[0])

print("\n========== DONE ==========\n")
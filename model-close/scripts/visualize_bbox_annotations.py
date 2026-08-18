import cv2
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "dfsp"
CSV_PATH = ROOT / "data" / "dfsp_bbox_annotations.csv"

rows = []

with open(
    CSV_PATH,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)
    rows = list(reader)


print("=" * 60)
print("DFSP BBOX ANNOTATION CHECK")
print("=" * 60)
print(f"Total annotations: {len(rows)}")
print()
print("Controls:")
print("ENTER = next image")
print("Q     = quit")
print("R     = previous image")
print("=" * 60)


index = 0

cv2.namedWindow(
    "BBOX CHECK",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "BBOX CHECK",
    1200,
    800
)


while 0 <= index < len(rows):

    row = rows[index]

    image_name = row["image"]

    image_path = IMAGE_DIR / image_name

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"❌ Cannot read: {image_name}"
        )

        index += 1
        continue

    x_min = int(float(row["x_min"]))
    y_min = int(float(row["y_min"]))
    x_max = int(float(row["x_max"]))
    y_max = int(float(row["y_max"]))

    center_x = float(row["center_x"])
    center_y = float(row["center_y"])

    # BBOX
    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2
    )

    # CENTER
    cv2.circle(
        image,
        (int(center_x), int(center_y)),
        5,
        (0, 0, 255),
        -1
    )

    # Bilgi
    text = (
        f"{index + 1}/{len(rows)} "
        f"| {row['split']} "
        f"| {image_name}"
    )

    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "BBOX CHECK",
        image
    )

    print(
        f"[{index + 1}/{len(rows)}] "
        f"{image_name} | "
        f"split={row['split']} | "
        f"center=({center_x:.1f}, {center_y:.1f})"
    )

    while True:

        key = cv2.waitKey(30) & 0xFF

        # ENTER
        if key == 13:

            index += 1
            break

        # Q
        elif key == ord("q"):

            cv2.destroyAllWindows()

            print("Check stopped.")

            raise SystemExit

        # R
        elif key == ord("r"):

            if index > 0:
                index -= 1

            break


cv2.destroyAllWindows()

print()
print("=" * 60)
print("✅ BBOX CHECK FINISHED")
print("=" * 60)
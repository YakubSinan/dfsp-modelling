import cv2
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "dfsp"
OUTPUT_CSV = ROOT / "data" / "dfsp_bbox_annotations.csv"

SPLITS = ["train", "val", "test"]

FIELDS = [
    "image",
    "label",
    "width",
    "height",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "center_x",
    "center_y",
    "split",
]


def load_dataset():

    items = []

    for split in SPLITS:

        csv_path = ROOT / "data" / f"dfsp_{split}.csv"

        with open(
            csv_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                items.append({
                    "image": row["image"],
                    "label": row.get("label", "DFSP"),
                    "split": split
                })

    return items


def load_annotations():

    data = {}

    if not OUTPUT_CSV.exists():
        return data

    with open(
        OUTPUT_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            data[row["image"]] = row

    return data


def save_annotations(data):

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()

        for row in data.values():
            writer.writerow(row)


items = load_dataset()
annotations = load_annotations()

missing = [
    item
    for item in items
    if item["image"] not in annotations
]

print()
print("=" * 60)
print("DFSP BBOX ANNOTATION")
print("=" * 60)
print(f"Toplam CSV görüntüsü : {len(items)}")
print(f"Mevcut annotation    : {len(annotations)}")
print(f"Eksik annotation     : {len(missing)}")
print("=" * 60)

if not missing:

    print("🎉 Tüm görüntüler zaten annotate edilmiş!")
    print("Program kapanıyor.")
    raise SystemExit


print("Eksik görüntüler:")

for item in missing:
    print("-", item["image"])

print()


# ---------------------------------------------------------
# Mouse
# ---------------------------------------------------------

drawing = False
start_x = 0
start_y = 0
box = None
image = None


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global box

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

        box = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:

        preview = image.copy()

        cv2.rectangle(
            preview,
            (start_x, start_y),
            (x, y),
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "DFSP Annotation",
            preview
        )

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        x_min = min(start_x, x)
        x_max = max(start_x, x)

        y_min = min(start_y, y)
        y_max = max(start_y, y)

        box = (
            x_min,
            y_min,
            x_max,
            y_max
        )


cv2.namedWindow(
    "DFSP Annotation",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "DFSP Annotation",
    1200,
    800
)

cv2.setMouseCallback(
    "DFSP Annotation",
    mouse_callback
)


# ---------------------------------------------------------
# Sadece eksik görüntüler
# ---------------------------------------------------------

for number, item in enumerate(missing, start=1):

    image_name = item["image"]

    print()
    print("=" * 60)
    print(f"EKSİK {number}/{len(missing)}")
    print(f"Dosya: {image_name}")
    print(f"Split: {item['split']}")
    print("=" * 60)

    image_path = IMAGE_DIR / image_name

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print("❌ Görüntü okunamadı:")
        print(image_path)
        print()

        continue

    box = None

    cv2.imshow(
        "DFSP Annotation",
        image
    )

    while True:

        key = cv2.waitKey(30) & 0xFF

        # Q = çık
        if key == ord("q"):

            save_annotations(annotations)

            cv2.destroyAllWindows()

            print()
            print("🛑 Çıkıldı.")
            print(
                f"Kaydedilmiş annotation: "
                f"{len(annotations)}"
            )

            raise SystemExit

        # R = tekrar
        elif key == ord("r"):

            box = None

            cv2.imshow(
                "DFSP Annotation",
                image
            )

        # ENTER = kaydet
        elif key == 13:

            if box is None:

                print(
                    "❌ Önce bbox çiz."
                )

                continue

            x_min, y_min, x_max, y_max = box

            height, width = image.shape[:2]

            center_x = (
                x_min + x_max
            ) / 2

            center_y = (
                y_min + y_max
            ) / 2

            annotations[image_name] = {

                "image": image_name,

                "label": item["label"],

                "width": width,

                "height": height,

                "x_min": x_min,

                "y_min": y_min,

                "x_max": x_max,

                "y_max": y_max,

                "center_x": center_x,

                "center_y": center_y,

                "split": item["split"],
            }

            save_annotations(annotations)

            print()
            print("✅ Kaydedildi.")
            print(
                f"Center: "
                f"({center_x:.1f}, "
                f"{center_y:.1f})"
            )

            break


cv2.destroyAllWindows()

print()
print("=" * 60)
print("🎉 EKSİK ANNOTATIONLAR TAMAMLANDI")
print("=" * 60)
print(
    f"Toplam annotation: {len(annotations)}"
)
from pathlib import Path
import csv
import cv2

ROOT = Path(__file__).resolve().parents[1]

DFSP_DIR = ROOT / "data" / "dfsp"
OUTPUT_CSV = ROOT / "data" / "annotations.csv"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".jfif",
    ".avif"
}


# Mevcut annotationları oku
existing_rows = []

with open(
    OUTPUT_CSV,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        existing_rows.append(row)


annotated_images = {
    row["image"]
    for row in existing_rows
    if row["label"] == "DFSP"
}


print(f"Mevcut annotation: {len(existing_rows)}")
print(f"Mevcut DFSP annotation: {len(annotated_images)}")


# Sadece annotation olmayan DFSP görüntülerini bul
missing_images = sorted(
    [
        p for p in DFSP_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.name not in annotated_images
    ]
)


print(f"DFSP klasöründeki görüntü: {sum(1 for p in DFSP_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)}")
print(f"Yeni DFSP görüntüsü: {len(missing_images)}")


if not missing_images:
    print("Yeni annotation yapılacak görüntü yok.")
    exit()


# Annotation
for index, image_path in enumerate(missing_images, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Okunamadı: {image_path.name}")
        continue

    height, width = image.shape[:2]

    scale = min(
        1000 / width,
        800 / height,
        1.0
    )

    display = cv2.resize(
        image,
        (
            int(width * scale),
            int(height * scale)
        )
    )

    clicked = {"point": None}

    window_name = f"DFSP {index}/{len(missing_images)}"

    def callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            original_x = int(x / scale)
            original_y = int(y / scale)

            clicked["point"] = (
                original_x,
                original_y
            )

            print(
                f"Selected: "
                f"({original_x}, {original_y})"
            )


    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        callback
    )


    while True:

        view = display.copy()

        cv2.putText(
            view,
            f"DFSP {index}/{len(missing_images)} | {image_path.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        if clicked["point"] is not None:

            x, y = clicked["point"]

            cv2.circle(
                view,
                (
                    int(x * scale),
                    int(y * scale)
                ),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                view,
                "ENTER = save | R = reset",
                (10, view.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                view,
                "Click lesion center",
                (10, view.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


        cv2.imshow(window_name, view)

        key = cv2.waitKey(30) & 0xFF


        # ENTER
        if key in (13, 10):

            if clicked["point"] is not None:

                x, y = clicked["point"]

                existing_rows.append({
                    "image": image_path.name,
                    "label": "DFSP",
                    "width": str(width),
                    "height": str(height),
                    "center_x": str(x),
                    "center_y": str(y)
                })

                print(
                    f"[SAVED] {image_path.name} "
                    f"-> ({x}, {y})"
                )

                break


        # R
        elif key in (ord("r"), ord("R")):

            clicked["point"] = None


        # ESC
        elif key == 27:

            cv2.destroyAllWindows()

            print("Annotation durduruldu.")
            exit()


    cv2.destroyAllWindows()


# Tüm annotationları kaydet
fieldnames = [
    "image",
    "label",
    "width",
    "height",
    "center_x",
    "center_y"
]

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(existing_rows)


print()
print("==============================")
print("Annotation tamamlandı!")
print(f"Toplam annotation: {len(existing_rows)}")
print("==============================")
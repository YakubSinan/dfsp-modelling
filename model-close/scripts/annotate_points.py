from pathlib import Path
import csv
import cv2

ROOT = Path(__file__).resolve().parents[1]

DATA_DIRS = {
    "DF": ROOT / "data" / "dermatofibroma",
    "DFSP": ROOT / "data" / "dfsp",
}

OUTPUT_CSV = ROOT / "data" / "annotations.csv"

image_extensions = {".jpg", ".jpeg", ".png"}

images = []

for label, folder in DATA_DIRS.items():
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in image_extensions:
            images.append((label, path))

print(f"Toplam görüntü: {len(images)}")
print(f"DF: {sum(1 for x in images if x[0] == 'DF')}")
print(f"DFSP: {sum(1 for x in images if x[0] == 'DFSP')}")

annotations = []

for index, (label, image_path) in enumerate(images, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[UYARI] Görüntü okunamadı: {image_path}")
        continue

    original = image.copy()

    h, w = image.shape[:2]

    scale = min(1000 / w, 800 / h, 1.0)

    display = cv2.resize(
        image,
        (int(w * scale), int(h * scale))
    )

    selected_point = None

    window_name = f"Annotation {index}/{len(images)} - {label}"



    clicked = {"point": None}

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(x / scale)
            original_y = int(y / scale)

            clicked["point"] = (original_x, original_y)

            print(
                f"  Selected center: "
                f"({original_x}, {original_y})"
            )

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback)

    while True:

        view = display.copy()

        cv2.putText(
            view,
            f"{label} | {image_path.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if clicked["point"] is not None:
            px, py = clicked["point"]

            display_x = int(px * scale)
            display_y = int(py * scale)

            cv2.circle(
                view,
                (display_x, display_y),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                view,
                "ENTER = save | R = reset",
                (10, view.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        else:
            cv2.putText(
                view,
                "Click lesion center",
                (10, view.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow(window_name, view)

        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):
            if clicked["point"] is not None:
                x, y = clicked["point"]

                annotations.append([
                    image_path.name,
                    label,
                    w,
                    h,
                    x,
                    y
                ])

                print(
                    f"[SAVED] {image_path.name} "
                    f"-> ({x}, {y})"
                )

                break

        # R -> tekrar seç
        elif key in (ord("r"), ord("R")):
            clicked["point"] = None
            print("  Point reset.")

        elif key == 27:
            print("\nAnnotation durduruldu.")
            cv2.destroyAllWindows()

            with open(
                OUTPUT_CSV,
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    "image",
                    "label",
                    "width",
                    "height",
                    "center_x",
                    "center_y"
                ])

                writer.writerows(annotations)

            print(
                f"Kaydedilen annotation: "
                f"{len(annotations)}"
            )

            raise SystemExit

    cv2.destroyAllWindows()


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "label",
        "width",
        "height",
        "center_x",
        "center_y"
    ])

    writer.writerows(annotations)

print("\nAnnotation tamamlandı!")
print(f"CSV: {OUTPUT_CSV}")
print(f"Toplam kaydedilen: {len(annotations)}")
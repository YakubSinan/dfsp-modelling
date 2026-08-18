import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "dfsp"

RENAMES = {
    "Ekran görüntüsü 2026-08-07 231647.jpg":
        "screenshot_231647.jpg",

    "Ekran görüntüsü 2026-08-07 231717.jpg":
        "screenshot_231717.jpg",

    "Ekran görüntüsü 2026-08-07 231729.jpg":
        "screenshot_231729.jpg",

    "Ekran görüntüsü 2026-08-07 233308.jpg":
        "screenshot_233308.jpg",

    "Ekran görüntüsü 2026-08-08 102449.jpg":
        "screenshot_102449.jpg",

    "Ekran görüntüsü 2026-08-08 102500.jpg":
        "screenshot_102500.jpg",
}


# ---------------------------------------------------------
# 1. Görüntü dosyalarını yeniden adlandır
# ---------------------------------------------------------

for old_name, new_name in RENAMES.items():

    old_path = IMAGE_DIR / old_name
    new_path = IMAGE_DIR / new_name

    if old_path.exists():

        old_path.rename(new_path)

        print(
            f"✅ {old_name}"
        )

        print(
            f"   → {new_name}"
        )

    elif new_path.exists():

        print(
            f"ℹ️ Zaten değiştirilmiş: {new_name}"
        )

    else:

        print(
            f"❌ Bulunamadı: {old_name}"
        )


# ---------------------------------------------------------
# 2. Train / Val / Test CSV'lerini güncelle
# ---------------------------------------------------------

for split in ["train", "val", "test"]:

    csv_path = ROOT / "data" / f"dfsp_{split}.csv"

    rows = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        for row in reader:

            if row["image"] in RENAMES:

                row["image"] = RENAMES[row["image"]]

            rows.append(row)


    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"✅ Güncellendi: dfsp_{split}.csv"
    )


print()
print("=" * 60)
print("🎉 İSİM DÜZELTME TAMAMLANDI")
print("=" * 60)
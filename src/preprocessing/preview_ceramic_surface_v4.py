
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageOps, ImageDraw
import csv
import random

ROOT = Path("data/external/ceramic_surface_v4")
OUTPUT = Path("reports/ceramic_surface_v4_preview.jpg")

SPLITS = ["train", "valid", "test"]
DAMAGE = {"crack", "scratch", "stain"}

groups = defaultdict(list)

for split in SPLITS:

    csv_path = ROOT / split / "_classes.csv"

    with csv_path.open(
        newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            filename = row["filename"].strip()
            image_path = ROOT / split / filename

            if not image_path.exists():
                print("Missing:", image_path)
                continue

            active = {
                cls
                for cls in [
                    "normal",
                    "crack",
                    "scratch",
                    "stain",
                    "Unlabeled"
                ]
                if row.get(cls, "").strip() == "1"
            }

            if active == {"normal"}:
                group = "normal"

            elif active == {"crack"}:
                group = "crack"

            elif active == {"scratch"}:
                group = "scratch"

            elif active == {"stain"}:
                group = "stain"

            elif len(active & DAMAGE) >= 2:
                group = "multiple_defects"

            else:
                group = "other"

            groups[group].append(
                (image_path, split, active)
            )


print("\n===== IMAGE CATEGORIES =====")

for group, items in sorted(groups.items()):
    print(f"{group:<20}: {len(items)}")


# Select 4 images from each useful category
rng = random.Random(42)

categories = [
    "normal",
    "crack",
    "scratch",
    "stain",
    "multiple_defects"
]

selected = []

for category in categories:

    candidates = groups[category]

    samples = rng.sample(
        candidates,
        min(4, len(candidates))
    )

    for sample in samples:
        selected.append((category, *sample))


# Create contact sheet
columns = 5
rows = 4

cell_w = 300
cell_h = 330

canvas = Image.new(
    "RGB",
    (columns * cell_w, rows * cell_h),
    "white"
)

for index, (category, path, split, labels) in enumerate(
    selected
):

    with Image.open(path) as original:

        image = ImageOps.contain(
            ImageOps.exif_transpose(
                original
            ).convert("RGB"),
            (280, 260)
        )

    tile = Image.new(
        "RGB",
        (cell_w, cell_h),
        "#eeeeee"
    )

    x = (cell_w - image.width) // 2
    y = 10 + (260 - image.height) // 2

    tile.paste(image, (x, y))

    draw = ImageDraw.Draw(tile)

    draw.text(
        (10, 277),
        f"{category} | {split}",
        fill="black"
    )

    draw.text(
        (10, 296),
        path.name[:34],
        fill="black"
    )

    canvas.paste(
        tile,
        (
            (index % columns) * cell_w,
            (index // columns) * cell_h
        )
    )


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

canvas.save(OUTPUT, quality=95)

print("\nPreview saved:", OUTPUT)
print("Images displayed:", len(selected))
print("Original dataset unchanged.")

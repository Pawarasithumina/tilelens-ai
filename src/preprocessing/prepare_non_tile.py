
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import random
import shutil
import csv
import math

SOURCE = Path(
    "data/external/caltech101/images/101_ObjectCategories"
)

OUTPUT = Path("data/curated/non_tile_caltech101")
PREVIEW = Path("reports/caltech_non_tile_preview.jpg")
MANIFEST = Path("reports/caltech_non_tile_manifest.csv")

CATEGORIES = [
    "camera",
    "cellphone",
    "chair",
    "grand_piano",
    "cup",
    "wrench",
    "headphone",
    "stop_sign",
    "laptop",
    "stapler",
    "scissors",
    "lamp",
    "watch",
    "ceiling_fan",
    "ewer",
    "umbrella",
    "barrel",
    "pizza"
]

MAX_PER_CATEGORY = 45
EXTENSIONS = {".jpg", ".jpeg", ".png"}

rng = random.Random(42)

# Check before creating files
if OUTPUT.exists():
    raise SystemExit(
        f"Output already exists: {OUTPUT}\n"
        "Review it before running again."
    )

for category in CATEGORIES:
    if not (SOURCE / category).is_dir():
        raise FileNotFoundError(
            f"Missing category: {category}"
        )

selected_images = []
preview_samples = []

print("\n===== NON-TILE DATASET PREPARATION =====")

for category in CATEGORIES:

    category_dir = SOURCE / category

    images = sorted(
        p for p in category_dir.iterdir()
        if p.suffix.lower() in EXTENSIONS
    )

    sample_size = min(
        MAX_PER_CATEGORY,
        len(images)
    )

    sampled = rng.sample(images, sample_size)

    destination = OUTPUT / category
    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for image_path in sampled:

        output_path = destination / image_path.name

        shutil.copy2(
            image_path,
            output_path
        )

        selected_images.append({
            "category": category,
            "source": str(image_path),
            "destination": str(output_path)
        })

    if sampled:
        preview_samples.append(
            (category, sampled[0])
        )

    print(
        f"{category:<18} "
        f"Available: {len(images):<5} "
        f"Selected: {sample_size}"
    )


# Save dataset provenance manifest
MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)

with MANIFEST.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "source",
            "destination"
        ]
    )

    writer.writeheader()
    writer.writerows(selected_images)


# Generate contact sheet
columns = 6
cell_width = 260
cell_height = 240

rows = math.ceil(
    len(preview_samples) / columns
)

canvas = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white"
)

for index, (category, image_path) in enumerate(
    preview_samples
):

    with Image.open(image_path) as original:
        image = ImageOps.contain(
            ImageOps.exif_transpose(original).convert("RGB"),
            (240, 195)
        )

    x = (index % columns) * cell_width
    y = (index // columns) * cell_height

    image_x = x + (cell_width - image.width) // 2
    image_y = y + 10

    canvas.paste(
        image,
        (image_x, image_y)
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (x + 10, y + 215),
        category,
        fill="black"
    )

PREVIEW.parent.mkdir(
    parents=True,
    exist_ok=True
)

canvas.save(PREVIEW, quality=95)

print("\n===== SUMMARY =====")
print("Categories:", len(CATEGORIES))
print("Selected images:", len(selected_images))
print("Dataset location:", OUTPUT)
print("Manifest:", MANIFEST)
print("Preview:", PREVIEW)
print("Original dataset unchanged.")
print("No train/validation/test split created.")

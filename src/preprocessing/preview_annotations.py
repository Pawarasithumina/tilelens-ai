
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import random

# Dataset paths
ROOT = Path("data/external/ceramic_defects")
IMAGE_DIR = ROOT / "train" / "images"
LABEL_DIR = ROOT / "train" / "labels"
OUTPUT = Path("reports/annotation_preview.jpg")

CLASS_NAMES = ["edge-chipping", "hole", "line"]
COLORS = ["red", "blue", "orange"]

random.seed(42)

# Find images
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

annotated = []
empty = []

for image_path in images:
    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if label_path.exists() and label_path.read_text().strip():
        annotated.append(image_path)
    else:
        empty.append(image_path)

# Sample 12 annotated and 4 empty-label images
selected = (
    random.sample(annotated, min(12, len(annotated)))
    + random.sample(empty, min(4, len(empty)))
)

random.shuffle(selected)

# Create 4x4 preview
cell_w, cell_h = 350, 330
canvas = Image.new("RGB", (cell_w * 4, cell_h * 4), "white")

for idx, image_path in enumerate(selected):

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    with Image.open(image_path) as original:
        image = ImageOps.contain(
            original.convert("RGB"),
            (320, 280)
        )

    tile = Image.new("RGB", (cell_w, cell_h), "#eeeeee")

    x_offset = (cell_w - image.width) // 2
    y_offset = 10 + (280 - image.height) // 2

    tile.paste(image, (x_offset, y_offset))
    draw = ImageDraw.Draw(tile)

    lines = (
        label_path.read_text().splitlines()
        if label_path.exists()
        else []
    )

    box_count = 0

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            continue

        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        if cls not in range(len(CLASS_NAMES)):
            continue

        x1 = x_offset + (xc - bw / 2) * image.width
        y1 = y_offset + (yc - bh / 2) * image.height
        x2 = x_offset + (xc + bw / 2) * image.width
        y2 = y_offset + (yc + bh / 2) * image.height

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=COLORS[cls],
            width=2
        )

        box_count += 1

    status = (
        "Empty label"
        if box_count == 0
        else f"{box_count} annotations"
    )

    draw.text(
        (10, 305),
        f"{image_path.name[:28]} | {status}",
        fill="black"
    )

    canvas.paste(
        tile,
        ((idx % 4) * cell_w, (idx // 4) * cell_h)
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT, quality=95)

print("Preview saved:", OUTPUT)
print("Annotated images available:", len(annotated))
print("Empty-label images available:", len(empty))
print("Colors: red=edge-chipping, blue=hole, orange=line")

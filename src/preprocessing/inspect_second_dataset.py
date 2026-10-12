
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageOps
import random
import math

ROOT = Path("data/external/tiles_defect_v5")
OUTPUT = Path("reports/second_dataset_preview.jpg")

CLASSES = [
    "background", "crack", "edge damage",
    "good", "holes", "pattern"
]

COLORS = [
    "#808080", "#ff4444", "#ff9900",
    "#00bb66", "#0088ff", "#aa44ff"
]

groups = {
    "good_only": [],
    "good_mixed": [],
    "invalid": []
}

errors = Counter()
examples = []
total_invalid = 0

def parse_annotation(line):
    parts = line.split()

    if len(parts) != 5:
        return None, "wrong number of fields"

    try:
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None, "non-numeric value"

    if cls < 0 or cls >= len(CLASSES):
        return None, "invalid class ID"

    if not all(math.isfinite(v) for v in coords):
        return None, "non-finite coordinate"

    if not all(0 <= v <= 1 for v in coords):
        return None, "coordinate outside 0-1"

    if coords[2] <= 0 or coords[3] <= 0:
        return None, "zero-size bounding box"

    return (cls, coords), None


for split in ["train", "valid", "test"]:
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in (
            ".jpg", ".jpeg", ".png"
        ):
            continue

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            continue

        valid_boxes = []
        invalid_count = 0

        for line_number, line in enumerate(
            label_path.read_text().splitlines(), start=1
        ):
            if not line.strip():
                continue

            annotation, error = parse_annotation(line)

            if error:
                errors[error] += 1
                total_invalid += 1
                invalid_count += 1

                if len(examples) < 12:
                    examples.append(
                        (split, image_path.name,
                         line_number, error, line[:90])
                    )
            else:
                valid_boxes.append(annotation)

        classes_present = {
            cls for cls, _ in valid_boxes
        }

        item = (split, image_path, valid_boxes)

        if invalid_count:
            groups["invalid"].append(item)

        if invalid_count == 0:
            if classes_present == {3}:
                groups["good_only"].append(item)

            elif 3 in classes_present and len(classes_present) > 1:
                groups["good_mixed"].append(item)


print("\n===== INVALID ANNOTATIONS =====")
print("Invalid annotation rows:", total_invalid)

for error, count in errors.items():
    print(f"{error}: {count}")

print("\nExamples:")
for split, filename, number, error, line in examples:
    print(f"\n{split}/{filename}")
    print(f"Line {number}: {error}")
    print(line)

print("\n===== IMAGE CATEGORIES =====")
for name, items in groups.items():
    print(f"{name}: {len(items)} images")


# Build visual preview
rng = random.Random(42)
selected = []

for category in ["good_only", "good_mixed", "invalid"]:
    items = groups[category]
    sample = rng.sample(items, min(4, len(items)))

    for item in sample:
        selected.append((category, *item))

cell_w, cell_h = 360, 310
cols = 4
rows = math.ceil(len(selected) / cols)

if rows == 0:
    print("No preview images found.")
    raise SystemExit

canvas = Image.new(
    "RGB", (cols * cell_w, rows * cell_h), "white"
)

for index, (category, split, path, boxes) in enumerate(selected):

    with Image.open(path) as original:
        image = ImageOps.contain(
            original.convert("RGB"), (330, 250)
        )

    tile = Image.new("RGB", (cell_w, cell_h), "#eeeeee")

    left = (cell_w - image.width) // 2
    top = 10 + (250 - image.height) // 2

    tile.paste(image, (left, top))
    draw = ImageDraw.Draw(tile)

    for cls, (xc, yc, bw, bh) in boxes:
        x1 = left + (xc - bw / 2) * image.width
        y1 = top + (yc - bh / 2) * image.height
        x2 = left + (xc + bw / 2) * image.width
        y2 = top + (yc + bh / 2) * image.height

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=COLORS[cls],
            width=2
        )

    draw.text(
        (10, 270),
        f"{category} | {split}",
        fill="black"
    )
    draw.text(
        (10, 287),
        path.name[:43],
        fill="black"
    )

    canvas.paste(
        tile,
        ((index % cols) * cell_w,
         (index // cols) * cell_h)
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT, quality=95)

print("\nPreview saved:", OUTPUT)
print("Bounding box colors:")
for name, color in zip(CLASSES, COLORS):
    print(name, color)

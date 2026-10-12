
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import math

ROOT = Path("data/external/ceramic_defects")
OUTPUT = Path("reports/non_dark_empty_labels.jpg")

selected = []

for split in ["train", "valid", "test"]:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    for image_path in sorted(image_dir.iterdir()):

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            continue

        if label_path.read_text().strip():
            continue

        with Image.open(image_path) as img:
            gray = img.convert("L")
            gray.thumbnail((128, 128))
            pixels = list(gray.get_flattened_data())

        dark_fraction = sum(
            pixel < 20 for pixel in pixels
        ) / len(pixels)

        if dark_fraction <= 0.95:
            selected.append((split, image_path, dark_fraction))

print("Non-dark empty-label images:", len(selected))

for split, path, fraction in selected:
    print(split, path.name, f"Dark pixels: {fraction:.1%}")

if selected:
    cols = 5
    rows = math.ceil(len(selected) / cols)

    cell_w = 320
    cell_h = 350

    canvas = Image.new(
        "RGB",
        (cols * cell_w, rows * cell_h),
        "white"
    )

    for i, (split, path, fraction) in enumerate(selected):

        with Image.open(path) as img:
            preview = ImageOps.contain(
                img.convert("RGB"),
                (300, 290)
            )

        x = (i % cols) * cell_w
        y = (i // cols) * cell_h

        canvas.paste(preview, (x + 10, y + 10))

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (x + 10, y + 310),
            f"{split}: {path.name[:32]}",
            fill="black"
        )

        draw.text(
            (x + 10, y + 328),
            f"Dark pixels: {fraction:.1%}",
            fill="black"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=95)

    print("\nSaved:", OUTPUT)
else:
    print("No non-dark empty-label images found.")

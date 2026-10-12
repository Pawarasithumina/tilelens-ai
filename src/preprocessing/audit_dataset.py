
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib

ROOT = Path("data/external/ceramic_defects")
SPLITS = ["train", "valid", "test"]

source_ids = defaultdict(set)
file_hashes = defaultdict(set)

for split in SPLITS:
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    empty_images = 0
    mostly_dark = 0

    for image_path in images:

        # Identify original Roboflow source filename
        source_id = image_path.stem.split(".rf.")[0]
        source_ids[source_id].add(split)

        # Detect byte-for-byte duplicates
        image_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        file_hashes[image_hash].add(split)

        label_path = label_dir / (
            image_path.stem + ".txt"
        )

        if not label_path.exists():
            continue

        if not label_path.read_text().strip():

            empty_images += 1

            with Image.open(image_path) as img:
                gray = img.convert("L")
                gray.thumbnail((128, 128))
                pixels = list(gray.getdata())

            dark_fraction = sum(
                p < 20 for p in pixels
            ) / len(pixels)

            if dark_fraction > 0.95:
                mostly_dark += 1

    print(f"\n===== {split.upper()} =====")
    print("Total images:", len(images))
    print("Empty-label images:", empty_images)
    print("Mostly-dark empty images:", mostly_dark)

# Check original image IDs across different splits
source_overlap = {
    k: v for k, v in source_ids.items()
    if len(v) > 1
}

# Check identical image files across splits
exact_overlap = {
    k: v for k, v in file_hashes.items()
    if len(v) > 1
}

print("\n===== DATA LEAKAGE AUDIT =====")
print("Unique source IDs:", len(source_ids))
print("Source IDs shared across splits:", len(source_overlap))
print("Exact image hashes shared across splits:", len(exact_overlap))

print("\nExamples of overlapping source IDs:")
for source_id, splits in list(source_overlap.items())[:10]:
    print(source_id, sorted(splits))


from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib

ROOT = Path("data/external/ceramic_defects")
SPLITS = ["train", "valid", "test"]

hash_groups = defaultdict(list)

# Collect image hashes
for split in SPLITS:
    image_dir = ROOT / split / "images"

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        image_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        hash_groups[image_hash].append(
            (split, image_path)
        )

# Find duplicate groups shared across splits
cross_split = {
    h: files
    for h, files in hash_groups.items()
    if len({split for split, _ in files}) > 1
}

print("Cross-split duplicate groups:", len(cross_split))

for i, (image_hash, files) in enumerate(
    cross_split.items(), start=1
):
    print(f"\n===== DUPLICATE GROUP {i} =====")
    print("Total matching files:", len(files))
    print("Hash:", image_hash[:16])

    for split in SPLITS:
        members = [
            p for s, p in files if s == split
        ]

        if not members:
            continue

        print(f"\n{split.upper()}: {len(members)} files")

        for path in members[:5]:
            label_path = (
                ROOT / split / "labels" /
                (path.stem + ".txt")
            )

            label_text = (
                label_path.read_text().strip()
                if label_path.exists()
                else ""
            )

            with Image.open(path) as img:
                gray = img.convert("L")
                gray.thumbnail((128, 128))

                pixels = list(
                    gray.get_flattened_data()
                )

                dark_fraction = sum(
                    value < 20 for value in pixels
                ) / len(pixels)

            print("  File:", path.name)
            print("  Empty label:", not bool(label_text))
            print(
                "  Dark pixels:",
                f"{dark_fraction * 100:.1f}%"
            )

        if len(members) > 5:
            print(
                f"  ...and {len(members) - 5} more"
            )

print("\nInspection completed.")

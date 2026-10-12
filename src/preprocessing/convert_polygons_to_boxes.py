
from pathlib import Path
from collections import Counter
import shutil
import math

SOURCE = Path("data/external/tiles_defect_v5")
OUTPUT = Path("data/processed/tiles_defect_v5_boxes")

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = [
    "background",
    "crack",
    "edge damage",
    "good",
    "holes",
    "pattern"
]

EXTENSIONS = {".jpg", ".jpeg", ".png"}

stats = Counter()
records = []


def convert_annotation(line):
    parts = line.strip().split()

    cls = int(parts[0])
    values = [float(v) for v in parts[1:]]

    if not 0 <= cls < len(CLASS_NAMES):
        raise ValueError("Invalid class ID")

    if not all(
        math.isfinite(v) and 0 <= v <= 1
        for v in values
    ):
        raise ValueError("Invalid coordinates")

    # Already a YOLO bounding box
    if len(values) == 4:
        xc, yc, w, h = values

        if w <= 0 or h <= 0:
            raise ValueError("Invalid box dimensions")

        stats["original_boxes"] += 1
        return line.strip()

    # YOLO segmentation polygon
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        width = x_max - x_min
        height = y_max - y_min

        if width <= 0 or height <= 0:
            raise ValueError("Invalid polygon dimensions")

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        stats["converted_polygons"] += 1

        return (
            f"{cls} "
            f"{x_center:.12g} "
            f"{y_center:.12g} "
            f"{width:.12g} "
            f"{height:.12g}"
        )

    raise ValueError("Unknown annotation format")


if OUTPUT.exists():
    raise SystemExit(
        f"Output already exists: {OUTPUT}\n"
        "Review the existing folder before rerunning."
    )


# Validate everything before copying
for split in SPLITS:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise RuntimeError(
                f"Missing label: {image_path.name}"
            )

        converted_lines = []

        for line_number, line in enumerate(
            label_path.read_text().splitlines(), start=1
        ):
            if not line.strip():
                continue

            try:
                converted_lines.append(
                    convert_annotation(line)
                )
            except (ValueError, IndexError) as error:
                raise RuntimeError(
                    f"{label_path}, line {line_number}: {error}"
                ) from error

        records.append(
            (split, image_path, converted_lines)
        )

        stats[f"{split}_images"] += 1


# Create a separate, standardized dataset
for split, image_path, converted_lines in records:
    image_output = OUTPUT / split / "images"
    label_output = OUTPUT / split / "labels"

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        image_path,
        image_output / image_path.name
    )

    label_path = label_output / f"{image_path.stem}.txt"

    label_path.write_text(
        "\n".join(converted_lines) + "\n"
        if converted_lines else ""
    )


# Write a dataset configuration with paths
# relative to the new dataset directory
yaml_text = (
    "train: train/images\n"
    "val: valid/images\n"
    "test: test/images\n"
    f"nc: {len(CLASS_NAMES)}\n"
    f"names: {CLASS_NAMES!r}\n"
)

(OUTPUT / "data.yaml").write_text(yaml_text)


print("\n===== CONVERSION SUMMARY =====")
print("Original bounding boxes:", stats["original_boxes"])
print("Converted polygons:", stats["converted_polygons"])

print("\n===== IMAGES =====")
for split in SPLITS:
    print(f"{split}: {stats[f'{split}_images']}")

print("\nTotal images:", sum(
    stats[f"{split}_images"] for split in SPLITS
))

print("\nOutput:", OUTPUT)
print("Original dataset preserved.")

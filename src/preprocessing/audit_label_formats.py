
from pathlib import Path
from collections import Counter
import math

ROOT = Path("data/external/tiles_defect_v5")

CLASSES = [
    "background",
    "crack",
    "edge damage",
    "good",
    "holes",
    "pattern"
]

DAMAGE_CLASSES = {1, 2, 4}
GOOD_CLASS = 3

format_counts = Counter()
polygon_classes = Counter()
image_categories = Counter()
invalid_examples = []

for split in ["train", "valid", "test"]:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    split_formats = Counter()

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    for image_path in images:

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            split_formats["missing_file"] += 1
            continue

        lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        if not lines:
            image_categories["empty_label"] += 1
            continue

        present_classes = set()
        image_has_invalid = False

        for line in lines:

            parts = line.split()
            kind = None

            try:
                cls = int(parts[0])
                values = [float(v) for v in parts[1:]]

                if not 0 <= cls < len(CLASSES):
                    raise ValueError("Invalid class ID")

                if not all(
                    math.isfinite(v) and 0 <= v <= 1
                    for v in values
                ):
                    raise ValueError("Invalid coordinates")

                if len(values) == 4:
                    xc, yc, w, h = values

                    if w <= 0 or h <= 0:
                        raise ValueError("Invalid box size")

                    kind = "bounding_box"

                elif (
                    len(values) >= 6
                    and len(values) % 2 == 0
                ):
                    xs = values[0::2]
                    ys = values[1::2]

                    if (
                        max(xs) <= min(xs)
                        or max(ys) <= min(ys)
                    ):
                        raise ValueError("Invalid polygon size")

                    kind = "segmentation_polygon"

                else:
                    raise ValueError("Unknown annotation format")

                present_classes.add(cls)

                if kind == "segmentation_polygon":
                    polygon_classes[cls] += 1

            except (ValueError, IndexError) as error:
                kind = "invalid"
                image_has_invalid = True

                if len(invalid_examples) < 10:
                    invalid_examples.append(
                        (split, image_path.name, str(error), line[:120])
                    )

            format_counts[kind] += 1
            split_formats[kind] += 1

        if image_has_invalid:
            image_categories["has_invalid_annotations"] += 1

        if GOOD_CLASS in present_classes:

            if present_classes == {GOOD_CLASS}:
                image_categories["good_only"] += 1

            elif present_classes & DAMAGE_CLASSES:
                image_categories["good_with_damage"] += 1

            else:
                image_categories["good_with_nondamage"] += 1

        else:
            image_categories["without_good"] += 1

    print(f"\n===== {split.upper()} =====")

    for kind, count in sorted(split_formats.items()):
        print(f"{kind}: {count}")


print("\n===== TOTAL ANNOTATION FORMATS =====")

for kind, count in sorted(format_counts.items()):
    print(f"{kind}: {count}")


print("\n===== POLYGON CLASSES =====")

for cls, count in sorted(polygon_classes.items()):
    print(f"{CLASSES[cls]}: {count}")


print("\n===== IMAGE LABEL COMBINATIONS =====")

for category, count in sorted(image_categories.items()):
    print(f"{category}: {count}")


print("\n===== INVALID EXAMPLES =====")

if not invalid_examples:
    print("No malformed annotations found.")

for split, name, error, line in invalid_examples:
    print(f"\n{split}/{name}")
    print("Reason:", error)
    print("Annotation:", line)

print("\nAudit completed. Original dataset unchanged.")


from pathlib import Path
from collections import Counter, defaultdict
import csv
import shutil
import math

ROOT = Path("data")

MANIFEST = ROOT / "processed/tile_classifier_v2/manifest.csv"

OUTPUT = ROOT / "processed/tile_damage_detector_v1"

CLASS_NAMES = [
    "edge-chipping",
    "hole",
    "line"
]

SPLITS = ["train", "val", "test"]


def convert_annotation(line, label_path, line_number):
    """
    Convert a YOLO bounding box or segmentation polygon
    into a YOLO bounding box:

    class_id x_center y_center width height
    """

    parts = line.split()

    cls = int(parts[0])

    if cls < 0 or cls >= len(CLASS_NAMES):
        raise ValueError(
            f"Invalid class {cls} in {label_path}:{line_number}"
        )

    values = [float(value) for value in parts[1:]]

    if not all(math.isfinite(v) for v in values):
        raise ValueError(
            f"Non-finite coordinates in {label_path}:{line_number}"
        )

    if not all(0 <= v <= 1 for v in values):
        raise ValueError(
            f"Coordinates outside [0,1] in "
            f"{label_path}:{line_number}"
        )

    is_polygon = False

    if len(values) == 4:
        # Standard YOLO detection annotation
        x_center, y_center, width, height = values

    elif len(values) >= 6 and len(values) % 2 == 0:
        # YOLO segmentation polygon:
        # x1 y1 x2 y2 x3 y3 ...

        xs = values[0::2]
        ys = values[1::2]

        x_min = min(xs)
        x_max = max(xs)

        y_min = min(ys)
        y_max = max(ys)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        width = x_max - x_min
        height = y_max - y_min

        is_polygon = True

    else:
        raise ValueError(
            f"Unsupported annotation format in "
            f"{label_path}:{line_number}, "
            f"number of values: {len(parts)}"
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Zero-size box in {label_path}:{line_number}"
        )

    converted = (
        f"{cls} "
        f"{x_center:.10f} "
        f"{y_center:.10f} "
        f"{width:.10f} "
        f"{height:.10f}"
    )

    return converted, cls, is_polygon


if OUTPUT.exists():
    raise SystemExit(
        f"{OUTPUT} already exists. Review it before rerunning."
    )

with MANIFEST.open(newline="", encoding="utf-8") as file:
    records = list(csv.DictReader(file))

selected = [
    row for row in records
    if row["origin"] == "ceramic_defects"
    and row["label"] == "tile"
]

print("===== SOURCE DATA =====")
print("Selected images:", len(selected))

stats = Counter()
class_counts = defaultdict(Counter)
group_splits = defaultdict(set)

polygon_count = 0
box_count = 0

prepared = []
conversion_log = []

# Validate and convert annotations before copying anything.
for row in selected:

    split = row["split"]

    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}")

    image_path = Path(row["source_path"])

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    label_path = (
        image_path.parent.parent
        / "labels"
        / f"{image_path.stem}.txt"
    )

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    annotations = []

    for line_number, line in enumerate(
        label_path.read_text().splitlines(),
        start=1
    ):

        if not line.strip():
            continue

        converted, cls, is_polygon = convert_annotation(
            line,
            label_path,
            line_number
        )

        annotations.append(converted)

        class_counts[split][cls] += 1

        if is_polygon:
            polygon_count += 1

            conversion_log.append({
                "source_file": str(label_path),
                "line_number": line_number,
                "original_annotation": line.strip(),
                "converted_annotation": converted
            })

        else:
            box_count += 1

    if not annotations:
        raise ValueError(
            f"Empty annotation file: {label_path}"
        )

    group_splits[row["group_id"]].add(split)

    prepared.append(
        (split, image_path, label_path, annotations)
    )


# Verify recognized source groups don't cross splits.
overlap = sum(
    len(s) > 1
    for s in group_splits.values()
)

if overlap:
    raise RuntimeError(
        "Known source groups cross dataset splits."
    )


# Create the new dataset.
for split, image_path, label_path, annotations in prepared:

    image_output = OUTPUT / split / "images"
    label_output = OUTPUT / split / "labels"

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    destination_image = image_output / image_path.name
    destination_label = label_output / label_path.name

    if destination_image.exists():
        raise RuntimeError(
            f"Duplicate destination: {destination_image}"
        )

    shutil.copy2(image_path, destination_image)

    # Write normalized bounding boxes, including converted polygons.
    destination_label.write_text(
        "\n".join(annotations) + "\n",
        encoding="utf-8"
    )

    stats[split] += 1


yaml_content = """train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: edge-chipping
  1: hole
  2: line
"""

(OUTPUT / "data.yaml").write_text(
    yaml_content,
    encoding="utf-8"
)


# Record polygon conversions for auditability.
if conversion_log:
    report_path = Path(
        "reports/damage_detector_polygon_conversions.csv"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(conversion_log[0].keys())
        )

        writer.writeheader()
        writer.writerows(conversion_log)


print("\n===== ANNOTATION SUMMARY =====")
print("Standard bounding boxes:", box_count)
print("Polygons converted:", polygon_count)
print("Total annotations:", box_count + polygon_count)

print("\n===== DATASET SUMMARY =====")

for split in SPLITS:

    print(f"\n{split.upper()}")
    print("Images:", stats[split])

    for cls, name in enumerate(CLASS_NAMES):
        print(
            f"  {name}: {class_counts[split][cls]}"
        )

print("\nTotal images:", sum(stats.values()))
print("Source-group overlap:", overlap)
print("Output:", OUTPUT)
print("Original datasets preserved.")

if conversion_log:
    print(
        "Conversion report: "
        "reports/damage_detector_polygon_conversions.csv"
    )

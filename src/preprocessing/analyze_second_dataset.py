
from pathlib import Path
from collections import Counter

ROOT = Path("data/external/tiles_defect_v5")

CLASSES = [
    "background",
    "crack",
    "edge damage",
    "good",
    "holes",
    "pattern"
]

SPLITS = ["train", "valid", "test"]

total_instances = Counter()
total_images_per_class = Counter()

for split in SPLITS:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    label_files = list(label_dir.glob("*.txt"))

    instance_counts = Counter()
    image_counts = Counter()

    empty_labels = 0
    missing_labels = 0
    invalid_annotations = 0
    good_only = 0
    good_with_other_classes = 0

    for image_path in images:

        label_path = label_dir / (
            image_path.stem + ".txt"
        )

        if not label_path.exists():
            missing_labels += 1
            continue

        lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1
            continue

        present_classes = set()

        for line in lines:

            parts = line.split()

            try:
                if len(parts) != 5:
                    raise ValueError("Invalid YOLO row")

                cls = int(parts[0])
                coords = list(map(float, parts[1:]))

                if cls not in range(len(CLASSES)):
                    raise ValueError("Invalid class ID")

                if not all(0 <= v <= 1 for v in coords):
                    raise ValueError("Invalid coordinates")

            except ValueError:
                invalid_annotations += 1
                continue

            instance_counts[cls] += 1
            present_classes.add(cls)

        for cls in present_classes:
            image_counts[cls] += 1

        if present_classes == {3}:
            good_only += 1

        if 3 in present_classes and len(present_classes) > 1:
            good_with_other_classes += 1

    print(f"\n{'=' * 45}")
    print(f"{split.upper()} DATASET")
    print(f"{'=' * 45}")

    print("Total images:", len(images))
    print("Label files:", len(label_files))
    print("Empty-label images:", empty_labels)
    print("Missing-label images:", missing_labels)
    print("Invalid annotation rows:", invalid_annotations)
    print("Good-only labeled images:", good_only)
    print(
        "Good with other classes:",
        good_with_other_classes
    )

    print("\nClass distribution:")
    print(f"{'Class':<16} {'Instances':>10} {'Images':>10}")

    for i, name in enumerate(CLASSES):
        print(
            f"{name:<16} "
            f"{instance_counts[i]:>10} "
            f"{image_counts[i]:>10}"
        )

        total_instances[i] += instance_counts[i]
        total_images_per_class[i] += image_counts[i]

print(f"\n{'=' * 45}")
print("OVERALL CLASS DISTRIBUTION")
print(f"{'=' * 45}")

print(f"{'Class':<16} {'Instances':>10} {'Images':>10}")

for i, name in enumerate(CLASSES):
    print(
        f"{name:<16} "
        f"{total_instances[i]:>10} "
        f"{total_images_per_class[i]:>10}"
    )

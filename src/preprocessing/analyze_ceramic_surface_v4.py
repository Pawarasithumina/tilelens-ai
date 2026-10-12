
from pathlib import Path
from collections import Counter
import csv

ROOT = Path("data/external/ceramic_surface_v4")
SPLITS = ["train", "valid", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

overall_counts = Counter()
total_images = 0

for split in SPLITS:
    folder = ROOT / split
    csv_path = folder / "_classes.csv"

    with csv_path.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    columns = [c.strip() for c in columns]

    filename_col = next(
        (c for c in columns if c.lower() == "filename"),
        None
    )

    if filename_col is None:
        raise ValueError(
            f"Cannot find filename column: {columns}"
        )

    classes = [
        c for c in columns if c != filename_col
    ]

    images = {
        p.name for p in folder.iterdir()
        if p.suffix.lower() in EXTENSIONS
    }

    counts = Counter()
    combinations = Counter()
    csv_filenames = set()
    zero_labels = 0
    multiple_labels = 0

    for row in rows:
        filename = Path(row[filename_col].strip()).name
        csv_filenames.add(filename)

        active = []

        for cls in classes:
            value = (row.get(cls) or "").strip().lower()

            if value in {"1", "true", "yes"}:
                counts[cls] += 1
                active.append(cls)

        if not active:
            zero_labels += 1

        if len(active) > 1:
            multiple_labels += 1

        combinations[tuple(sorted(active))] += 1

    print(f"\n{'=' * 50}")
    print(f"{split.upper()} DATASET")
    print(f"{'=' * 50}")

    print("Images on disk:", len(images))
    print("CSV records:", len(rows))
    print("Class columns:", classes)
    print("Unlabeled records:", zero_labels)
    print("Multi-label records:", multiple_labels)
    print("Images missing from CSV:", len(images - csv_filenames))
    print("CSV entries without images:", len(csv_filenames - images))

    print("\nClass distribution:")
    for cls in classes:
        print(f"  {cls:<25} {counts[cls]}")
        overall_counts[cls] += counts[cls]

    print("\nMost common label combinations:")
    for combo, count in combinations.most_common(10):
        print(f"  {combo}: {count}")

    total_images += len(images)

print(f"\n{'=' * 50}")
print("OVERALL SUMMARY")
print(f"{'=' * 50}")

print("Total images:", total_images)

for cls, count in overall_counts.most_common():
    print(f"{cls:<25} {count}")

print("\nAudit completed. No files modified.")


from pathlib import Path
from collections import Counter
import csv

REPORT = Path(
    "reports/verified_duplicate_candidates.csv"
)

CLASS_1 = {
    0: "edge-chipping",
    1: "hole",
    2: "line"
}

CLASS_2 = {
    0: "background",
    1: "crack",
    2: "edge damage",
    3: "good",
    4: "holes",
    5: "pattern"
}

DAMAGE_2 = {1, 2, 4}


def read_classes(image_path):
    image_path = Path(image_path)

    label_path = (
        image_path.parent.parent
        / "labels"
        / (image_path.stem + ".txt")
    )

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    classes = set()

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        cls = int(line.split()[0])
        classes.add(cls)

    return classes


with REPORT.open(newline="") as f:
    records = list(csv.DictReader(f))

strong = [
    r for r in records
    if r["verification"] == "strong_candidate"
]

stats = Counter()
conflicts = []

for record in strong:
    a = read_classes(record["dataset_1_image"])
    b = read_classes(record["dataset_2_image"])

    cross_split = (
        record["dataset_1_split"]
        != record["dataset_2_split"]
    )

    if cross_split:
        stats["cross_split"] += 1

    if 3 in b:
        stats["contains_good"] += 1

    if b & DAMAGE_2:
        stats["contains_damage_2"] += 1

    # A defect-annotated image is also marked good
    # without an explicit damage annotation in Dataset 2.
    if a and 3 in b and not (b & DAMAGE_2):
        stats["potential_disagreement"] += 1

        if cross_split:
            stats["cross_split_disagreement"] += 1

        conflicts.append((record, a, b))


print("\n===== LABEL COMPARISON =====")
print("Strong similarity pairs:", len(strong))
print("Cross-split pairs:", stats["cross_split"])

print(
    "Pairs containing Dataset 2 good:",
    stats["contains_good"]
)

print(
    "Pairs containing Dataset 2 damage:",
    stats["contains_damage_2"]
)

print(
    "Potential label disagreements:",
    stats["potential_disagreement"]
)

print(
    "Cross-split potential disagreements:",
    stats["cross_split_disagreement"]
)

print("\n===== EXAMPLES =====")

for record, a, b in sorted(
    conflicts,
    key=lambda item: -float(item[0]["ssim"])
)[:10]:

    print("\nSSIM:", record["ssim"])

    print(
        "Dataset 1:",
        record["dataset_1_split"],
        sorted(CLASS_1[c] for c in a)
    )

    print(
        "Dataset 2:",
        record["dataset_2_split"],
        sorted(CLASS_2[c] for c in b)
    )

    print(
        "Image 1:",
        Path(record["dataset_1_image"]).name
    )

    print(
        "Image 2:",
        Path(record["dataset_2_image"]).name
    )

print("\nAudit completed.")
print("No files modified.")

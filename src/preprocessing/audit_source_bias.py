
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import numpy as np
import csv
import re

ROOT = Path("data/external/ceramic_surface_v4")
EARLIER = Path("data/processed/ceramic_defects_v1")
SPLITS = ["train", "valid", "test"]

records = []
groups = defaultdict(list)



def source_info(filename):

    original = filename.split(".rf.")[0]

    # IMG camera sequence: check this first.
    # Group images from the same capture session.
    match = re.search(
        r"(?:^|_)IMG_(\d+)_(\d+)(?:_|$)",
        original,
        re.IGNORECASE
    )

    if match:
        return (
            "IMG_series",
            "IMG_SESSION_" + match.group(1)
        )

    # Actual MG camera identifiers.
    # Do not accidentally match IMG filenames.
    match = re.search(
        r"(?:^|_)MG_(\d+)(?:_|$)",
        original,
        re.IGNORECASE
    )

    if match:
        return (
            "MG_camera",
            "MG_" + match.group(1)
        )

    # Q-series identifiers.
    # Group by item number as a conservative
    # possible shared-source grouping.
    match = re.search(
        r"(?:^|[_-])(\d+)-Q\d+-[A-Z]",
        original,
        re.IGNORECASE
    )

    if match:
        return (
            "Q_series",
            "QITEM_" + match.group(1)
        )

    if "captured_image" in original.lower():
        return "captured_image", None

    return "other", None



def border_dark_fraction(path):
    with Image.open(path) as img:
        gray = img.convert("L").resize((96, 96))
        pixels = np.asarray(gray)

    border = np.concatenate([
        pixels[:5, :].ravel(),
        pixels[-5:, :].ravel(),
        pixels[:, :5].ravel(),
        pixels[:, -5:].ravel()
    ])

    return float(np.mean(border < 45))


for split in SPLITS:
    csv_path = ROOT / split / "_classes.csv"

    with csv_path.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        filename = row["filename"].strip()
        path = ROOT / split / filename

        active = {
            cls for cls in [
                "normal", "crack", "scratch",
                "stain", "Unlabeled"
            ]
            if row.get(cls, "").strip() == "1"
        }

        if active == {"normal"}:
            status = "normal"
        elif active & {"crack", "scratch", "stain"}:
            status = "damage"
        else:
            status = "other"

        family, group_id = source_info(filename)

        record = {
            "split": split,
            "filename": filename,
            "status": status,
            "family": family,
            "group_id": group_id,
            "dark_border": border_dark_fraction(path)
        }

        records.append(record)

        if group_id:
            groups[group_id].append(record)


print("\n===== SOURCE DISTRIBUTION =====")

for status in ["normal", "damage", "other"]:
    subset = [
        r for r in records if r["status"] == status
    ]

    families = Counter(r["family"] for r in subset)

    print(f"\n{status.upper()}: {len(subset)} images")

    for family, count in families.most_common():
        print(f"  {family:<20} {count}")

    if subset:
        average = np.mean([
            r["dark_border"] for r in subset
        ])

        prominent = sum(
            r["dark_border"] >= 0.25 for r in subset
        )

        print(f"  Average dark-border fraction: {average:.3f}")
        print(f"  Images with prominent dark border: {prominent}")


# Possible related images in different splits
overlaps = {}

for group_id, items in groups.items():
    splits = {r["split"] for r in items}

    if len(splits) > 1:
        overlaps[group_id] = items

print("\n===== POSSIBLE SOURCE-GROUP LEAKAGE =====")
print("Groups spanning multiple splits:", len(overlaps))

for group_id, items in list(overlaps.items())[:15]:
    print(
        group_id,
        "images:", len(items),
        "splits:", sorted({r["split"] for r in items}),
        "labels:", sorted({r["status"] for r in items})
    )


# Check filename-derived MG source IDs
# against our earlier defect dataset
earlier_sources = defaultdict(set)

for split in SPLITS:
    image_dir = EARLIER / split / "images"

    for path in image_dir.iterdir():
        if path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        family, group_id = source_info(path.name)

        if family == "MG_camera":
            earlier_sources[group_id].add(split)


matching = [
    r for r in records
    if r["group_id"] in earlier_sources
]

cross_split = [
    r for r in matching
    if any(
        s != r["split"]
        for s in earlier_sources[r["group_id"]]
    )
]

print("\n===== OVERLAP WITH DATASET 1 =====")
print("Matching MG-derived image records:", len(matching))
print(
    "Matches involving different splits:",
    len(cross_split)
)

print("\nLabel types among matches:")
print(Counter(r["status"] for r in matching))

print("\nIMPORTANT:")
print("Source IDs are filename-based clues, not proof.")
print("Visual duplicate verification is still required.")
print("Original datasets remain unchanged.")

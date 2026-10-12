
from pathlib import Path
from collections import defaultdict
import shutil
import re

SOURCE = Path("data/external/ceramic_defects")
OUTPUT = Path("data/processed/ceramic_defects_v1")

SPLITS = ["train", "valid", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

selected = {}
source_groups = defaultdict(set)

# Step 1: Identify annotated images
for split in SPLITS:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    selected[split] = []

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise RuntimeError(
                f"Missing annotation file: {image_path}"
            )

        if not label_path.read_text().strip():
            continue

        selected[split].append((image_path, label_path))

        # Identify original camera photograph
        match = re.search(
            r"(?:^|_)MG_(\d+)(?:_|$)",
            image_path.stem,
            re.IGNORECASE
        )

        if match:
            source_id = "MG_" + match.group(1)
        else:
            source_id = image_path.stem.split(".rf.")[0]

        source_groups[source_id].add(split)

# Step 2: Check source leakage
overlaps = {
    key: splits
    for key, splits in source_groups.items()
    if len(splits) > 1
}

print("Source groups shared across splits:", len(overlaps))

if overlaps:
    for name, splits in list(overlaps.items())[:20]:
        print(name, sorted(splits))

    raise SystemExit(
        "Potential source leakage detected. "
        "Review these groups before preparing data."
    )

# Step 3: Preserve existing output
if OUTPUT.exists():
    raise SystemExit(
        f"Output folder already exists: {OUTPUT}. "
        "Review it before rerunning."
    )

# Step 4: Copy annotated images and labels
for split in SPLITS:
    output_images = OUTPUT / split / "images"
    output_labels = OUTPUT / split / "labels"

    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)

    for image_path, label_path in selected[split]:
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        shutil.copy2(
            label_path,
            output_labels / label_path.name
        )

    print(f"{split}: {len(selected[split])} images")

print("\nDataset prepared:", OUTPUT)
print("Original dataset remains unchanged.")

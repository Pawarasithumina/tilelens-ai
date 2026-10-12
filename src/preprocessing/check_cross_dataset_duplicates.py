
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageOps, ImageDraw
import imagehash
import hashlib
import csv

DATASETS = {
    "ceramic": Path("data/processed/ceramic_defects_v1"),
    "tiles": Path("data/processed/tiles_defect_v5_boxes"),
}

REPORT = Path("reports")
REPORT.mkdir(exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png"}

def load_dataset(name, root):
    records = []

    for split in ["train", "valid", "test"]:
        image_dir = root / split / "images"

        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue

            exact_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            with Image.open(path) as img:
                perceptual_hash = int(
                    str(imagehash.phash(img.convert("RGB"))),
                    16
                )

            records.append({
                "dataset": name,
                "split": split,
                "path": path,
                "sha": exact_hash,
                "phash": perceptual_hash
            })

    return records


print("Loading first dataset...")
a_images = load_dataset("ceramic", DATASETS["ceramic"])

print("Loading second dataset...")
b_images = load_dataset("tiles", DATASETS["tiles"])

print("Dataset 1 images:", len(a_images))
print("Dataset 2 images:", len(b_images))

# Exact duplicates
b_by_sha = defaultdict(list)

for item in b_images:
    b_by_sha[item["sha"]].append(item)

exact = []

for a in a_images:
    for b in b_by_sha.get(a["sha"], []):
        exact.append(("exact", 0, a, b))

# Perceptually similar image candidates
# Keep at most two closest matches per first-dataset image.
near = []
threshold = 4

for a in a_images:
    candidates = []

    for b in b_images:
        if a["sha"] == b["sha"]:
            continue

        distance = (
            a["phash"] ^ b["phash"]
        ).bit_count()

        if distance <= threshold:
            candidates.append((distance, b))

    candidates.sort(key=lambda x: x[0])

    for distance, b in candidates[:2]:
        near.append(("possible", distance, a, b))

pairs = exact + near

# Save results
csv_path = REPORT / "cross_dataset_duplicates.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "match_type", "phash_distance",
        "dataset_1_split", "dataset_1_image",
        "dataset_2_split", "dataset_2_image",
        "different_split"
    ])

    for kind, distance, a, b in pairs:
        writer.writerow([
            kind, distance,
            a["split"], str(a["path"]),
            b["split"], str(b["path"]),
            a["split"] != b["split"]
        ])

print("\n===== DUPLICATE AUDIT =====")
print("Exact duplicate pairs:", len(exact))
print("Perceptual candidate pairs:", len(near))
print("Candidate pairs reported:", len(pairs))

cross_split = sum(
    a["split"] != b["split"]
    for _, _, a, b in pairs
)

print("Pairs in different splits:", cross_split)
print("CSV saved:", csv_path)

# Visualize the first 8 candidate pairs
preview_pairs = (
    exact[:4]
    + sorted(near, key=lambda x: x[1])[:8]
)[:8]

if preview_pairs:
    cell_width = 900
    cell_height = 285

    canvas = Image.new(
        "RGB",
        (cell_width, len(preview_pairs) * cell_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for i, (kind, distance, a, b) in enumerate(preview_pairs):
        y = i * cell_height

        for j, item in enumerate([a, b]):
            with Image.open(item["path"]) as img:
                image = ImageOps.contain(
                    img.convert("RGB"),
                    (420, 230)
                )

            x = j * 450 + (450 - image.width) // 2

            canvas.paste(image, (x, y + 20))

            draw.text(
                (j * 450 + 10, y + 253),
                f'{item["dataset"]} / {item["split"]}',
                fill="black"
            )

        draw.text(
            (10, y + 3),
            f"{kind} match | pHash distance: {distance}",
            fill="black"
        )

    preview_path = REPORT / "cross_dataset_preview.jpg"
    canvas.save(preview_path, quality=95)

    print("Preview saved:", preview_path)
else:
    print("No candidate pairs found at this threshold.")

print("\nAudit completed.")
print("No dataset files were modified.")

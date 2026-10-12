
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageOps, ImageDraw
from skimage.metrics import structural_similarity
import imagehash
import numpy as np
import csv
import random

ROOT = Path("data/processed/tile_classifier_v1")
REPORT = Path("reports")
REPORT.mkdir(parents=True, exist_ok=True)

with (ROOT / "manifest.csv").open(newline="") as f:
    records = list(csv.DictReader(f))

print("===== DATASET SUMMARY =====")
print("Total images:", len(records))

for split in ["train", "val", "test"]:
    subset = [r for r in records if r["split"] == split]

    print(f"\n{split.upper()}: {len(subset)}")
    print("Classes:", dict(Counter(r["label"] for r in subset)))
    print("Sources:", dict(Counter(r["origin"] for r in subset)))

# Check source-group and exact-hash overlap
groups = defaultdict(set)
hashes = defaultdict(set)

for r in records:
    groups[r["group_id"]].add(r["split"])
    hashes[r["sha256"]].add(r["split"])

print("\n===== LEAKAGE CHECK =====")
print(
    "Source groups spanning splits:",
    sum(len(v) > 1 for v in groups.values())
)
print(
    "Exact hashes spanning splits:",
    sum(len(v) > 1 for v in hashes.values())
)

# Perceptual similarity screening
print("\nComputing image perceptual hashes...")

for i, r in enumerate(records, start=1):
    path = Path(r["destination"])

    with Image.open(path) as img:
        r["phash"] = int(str(imagehash.phash(img)), 16)

    if i % 500 == 0:
        print(f"Processed {i}/{len(records)}")


def load_gray(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("L")
        img = img.resize((128, 128))
        return np.asarray(img, dtype=np.uint8)


train = [r for r in records if r["split"] == "train"]
val = [r for r in records if r["split"] == "val"]
test = [r for r in records if r["split"] == "test"]

candidates = []

# Compare validation against training,
# and test against both other splits.
for target, reference in [
    (val, train),
    (test, train + val)
]:
    for r in target:
        nearest = min(
            reference,
            key=lambda x: (
                r["phash"] ^ x["phash"]
            ).bit_count()
        )

        distance = (
            r["phash"] ^ nearest["phash"]
        ).bit_count()

        if distance <= 4:
            a = load_gray(r["destination"])
            b = load_gray(nearest["destination"])

            score = structural_similarity(
                a, b, data_range=255
            )

            candidates.append(
                (score, distance, r, nearest)
            )

candidates.sort(key=lambda x: x[0], reverse=True)

print("\n===== SIMILARITY SCREENING =====")
print("Near-hash candidates:", len(candidates))
print(
    "High-SSIM candidates (>= 0.94):",
    sum(x[0] >= 0.94 for x in candidates)
)

print("\nTop candidate matches:")
for score, dist, a, b in candidates[:10]:
    print(
        f"SSIM={score:.4f}, pHash={dist}, "
        f"{a['split']}/{a['origin']} vs "
        f"{b['split']}/{b['origin']}"
    )

# Generate representative image preview
rng = random.Random(42)
selected = []

for split in ["train", "val", "test"]:
    for label, origin in [
        ("tile", "ceramic_defects"),
        ("tile", "ceramic_surface"),
        ("not_tile", "caltech")
    ]:
        pool = [
            r for r in records
            if r["split"] == split
            and r["label"] == label
            and r["origin"] == origin
        ]

        selected.extend(rng.sample(pool, min(2, len(pool))))

cols = 6
cell_w, cell_h = 260, 245
rows = (len(selected) + cols - 1) // cols

canvas = Image.new(
    "RGB",
    (cols * cell_w, rows * cell_h),
    "white"
)

for i, r in enumerate(selected):
    with Image.open(r["destination"]) as img:
        preview = ImageOps.contain(
            ImageOps.exif_transpose(img).convert("RGB"),
            (240, 195)
        )

    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    canvas.paste(preview, (x + 10, y + 8))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (x + 10, y + 210),
        f"{r['split']} | {r['label']}",
        fill="black"
    )
    draw.text(
        (x + 10, y + 226),
        r["origin"],
        fill="black"
    )

output = REPORT / "classifier_v1_preview.jpg"
canvas.save(output, quality=95)

print("\nPreview saved:", output)
print("Original images unchanged.")



# ==========================================
# HIGH-SIMILARITY REVIEW REPORT
# ==========================================

high_similarity = [
    item for item in candidates
    if item[0] >= 0.94
]

# Save all first-pass candidate pairs
csv_output = REPORT / "classifier_similarity_candidates.csv"

with csv_output.open("w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "ssim",
        "phash_distance",
        "image_a",
        "split_a",
        "origin_a",
        "image_b",
        "split_b",
        "origin_b"
    ])

    for score, distance, a, b in candidates:
        writer.writerow([
            round(score, 6),
            distance,
            a["destination"],
            a["split"],
            a["origin"],
            b["destination"],
            b["split"],
            b["origin"]
        ])


# Generate contact sheets, 10 pairs per sheet
pairs_per_sheet = 10

for start in range(
    0, len(high_similarity), pairs_per_sheet
):

    batch = high_similarity[
        start:start + pairs_per_sheet
    ]

    row_h = 305
    canvas = Image.new(
        "RGB",
        (940, row_h * len(batch)),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for index, (score, distance, a, b) in enumerate(batch):
        y = index * row_h

        draw.text(
            (10, y + 5),
            f"SSIM: {score:.4f} | pHash: {distance}",
            fill="black"
        )

        for column, record in enumerate([a, b]):
            path = Path(record["destination"])

            with Image.open(path) as original:
                preview = ImageOps.contain(
                    ImageOps.exif_transpose(
                        original
                    ).convert("RGB"),
                    (440, 225)
                )

            left = column * 470
            image_x = left + (470 - preview.width) // 2

            canvas.paste(
                preview,
                (image_x, y + 27)
            )

            draw.text(
                (left + 10, y + 257),
                f"{record['split']} | {record['origin']}",
                fill="black"
            )

            draw.text(
                (left + 10, y + 278),
                path.name[:58],
                fill="black"
            )

    sheet_number = start // pairs_per_sheet + 1

    output = REPORT / (
        f"classifier_high_similarity_{sheet_number:02d}.jpg"
    )

    canvas.save(output, quality=92)
    print("Review sheet saved:", output)


print("\n===== SIMILARITY REVIEW =====")
print("All candidate pairs:", len(candidates))
print("Pairs with SSIM >= 0.94:", len(high_similarity))
print("CSV report:", csv_output)
print("No dataset files modified.")

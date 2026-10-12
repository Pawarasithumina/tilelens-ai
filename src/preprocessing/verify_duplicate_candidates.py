
from pathlib import Path
from functools import lru_cache
from collections import Counter
import csv

import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

INPUT = Path("reports/cross_dataset_duplicates.csv")
OUTPUT = Path("reports/verified_duplicate_candidates.csv")

IMAGE_SIZE = (192, 192)


@lru_cache(maxsize=3500)
def load_image(filename):
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = img.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        return np.asarray(img, dtype=np.uint8)


def compare_images(path_a, path_b):
    a = load_image(path_a)
    b = load_image(path_b)

    # Structural similarity
    score = structural_similarity(
        a, b, data_range=255
    )

    # Pixel correlation
    x = a.astype(np.float32).ravel()
    y = b.astype(np.float32).ravel()

    x -= x.mean()
    y -= y.mean()

    denominator = (
        np.linalg.norm(x) * np.linalg.norm(y)
    )

    if denominator == 0:
        correlation = (
            1.0 if np.array_equal(a, b) else 0.0
        )
    else:
        correlation = float(
            np.dot(x, y) / denominator
        )

    return float(score), correlation


def categorize(score, correlation):
    # Conservative screening criteria;
    # these are NOT proof of duplication.
    if score >= 0.94 and correlation >= 0.97:
        return "strong_candidate"

    if score >= 0.80 and correlation >= 0.90:
        return "needs_review"

    return "low_similarity"


with INPUT.open(newline="") as f:
    records = list(csv.DictReader(f))

print("Candidate pairs:", len(records))
print("Comparing images...")

results = []
counts = Counter()
cross_split_strong = []
unique_strong_images = set()

for index, record in enumerate(records, start=1):
    path_a = record["dataset_1_image"]
    path_b = record["dataset_2_image"]

    score, correlation = compare_images(
        path_a, path_b
    )

    category = categorize(
        score, correlation
    )

    different_split = (
        record["dataset_1_split"]
        != record["dataset_2_split"]
    )

    updated = {
        **record,
        "ssim": round(score, 5),
        "correlation": round(correlation, 5),
        "verification": category
    }

    results.append(updated)
    counts[category] += 1

    if category == "strong_candidate":
        unique_strong_images.update(
            [path_a, path_b]
        )

        if different_split:
            cross_split_strong.append(updated)

    if index % 250 == 0:
        print(f"Processed {index}/{len(records)}")


# Write verification report
if results:
    fieldnames = list(results[0].keys())

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)


print("\n===== VERIFICATION SUMMARY =====")
print("Total candidate pairs:", len(records))
print(
    "Strong visual candidates:",
    counts["strong_candidate"]
)
print(
    "Needs manual review:",
    counts["needs_review"]
)
print(
    "Lower similarity:",
    counts["low_similarity"]
)

print(
    "Strong pairs across splits:",
    len(cross_split_strong)
)
print(
    "Unique images in strong pairs:",
    len(unique_strong_images)
)

print("\n===== CROSS-SPLIT EXAMPLES =====")

for record in sorted(
    cross_split_strong,
    key=lambda r: -float(r["ssim"])
)[:10]:

    print(
        f"SSIM={record['ssim']} "
        f"Correlation={record['correlation']}"
    )

    print(
        "  Dataset 1:",
        record["dataset_1_split"],
        Path(record["dataset_1_image"]).name
    )

    print(
        "  Dataset 2:",
        record["dataset_2_split"],
        Path(record["dataset_2_image"]).name
    )

print("\nReport saved:", OUTPUT)
print("No original images or labels modified.")

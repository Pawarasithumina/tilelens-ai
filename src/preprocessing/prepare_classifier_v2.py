
from pathlib import Path
from collections import Counter, defaultdict
import csv
import shutil

SOURCE = Path("data/processed/tile_classifier_v1")
OUTPUT = Path("data/processed/tile_classifier_v2")

SIMILARITY_CSV = Path(
    "reports/classifier_similarity_candidates.csv"
)

QUARANTINE_REPORT = Path(
    "reports/classifier_v2_exclusions.csv"
)

THRESHOLD = 0.94
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if OUTPUT.exists():
    raise SystemExit(
        f"{OUTPUT} already exists. "
        "Review it before rerunning."
    )

manifest = read_csv(SOURCE / "manifest.csv")
candidates = read_csv(SIMILARITY_CSV)

# Paths of suspicious evaluation images
quarantine = {}

strong_pairs = [
    row for row in candidates
    if float(row["ssim"]) >= THRESHOLD
]

for row in strong_pairs:

    a = Path(row["image_a"]).resolve()
    b = Path(row["image_b"]).resolve()

    split_a = row["split_a"]
    split_b = row["split_b"]

    if split_a == split_b:
        continue

    # Keep the image in the earlier split.
    # Exclude its suspected match from the later split.
    if SPLIT_PRIORITY[split_a] > SPLIT_PRIORITY[split_b]:
        excluded, retained = a, b
    else:
        excluded, retained = b, a

    quarantine[str(excluded)] = {
        "excluded_image": str(excluded),
        "matched_image": str(retained),
        "ssim": row["ssim"],
        "reason": "high_similarity_cross_split"
    }


# Verify excluded images are from the manifest
all_paths = {
    str(Path(r["destination"]).resolve())
    for r in manifest
}

unknown = set(quarantine) - all_paths

if unknown:
    raise RuntimeError(
        f"Some exclusion paths are missing: {len(unknown)}"
    )

print("===== CANDIDATE REPAIR =====")
print("Strong similarity pairs:", len(strong_pairs))
print("Unique images marked for exclusion:", len(quarantine))

# Guard against unexpectedly large exclusions
evaluation_count = sum(
    r["split"] in {"val", "test"}
    for r in manifest
)

if len(quarantine) > evaluation_count * 0.20:
    raise SystemExit(
        "More than 20% of evaluation images flagged. "
        "Manual review required before proceeding."
    )


# Create revised dataset without changing v1
new_manifest = []
excluded_records = []

for record in manifest:

    old_path = Path(record["destination"])
    absolute_path = str(old_path.resolve())

    if absolute_path in quarantine:
        excluded_records.append({
            **record,
            **quarantine[absolute_path]
        })
        continue

    relative = old_path.relative_to(SOURCE)
    new_path = OUTPUT / relative

    new_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(old_path, new_path)

    new_record = {
        **record,
        "destination": str(new_path)
    }

    new_manifest.append(new_record)


# Write new manifest
manifest_path = OUTPUT / "manifest.csv"

with manifest_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(new_manifest[0].keys())
    )
    writer.writeheader()
    writer.writerows(new_manifest)


# Record all excluded images
if excluded_records:
    with QUARANTINE_REPORT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(excluded_records[0].keys())
        )
        writer.writeheader()
        writer.writerows(excluded_records)


# Verify exact hashes and source groups
hash_splits = defaultdict(set)
group_splits = defaultdict(set)

for record in new_manifest:
    hash_splits[record["sha256"]].add(record["split"])
    group_splits[record["group_id"]].add(record["split"])

exact_leakage = sum(
    len(s) > 1 for s in hash_splits.values()
)

group_leakage = sum(
    len(s) > 1 for s in group_splits.values()
)

if exact_leakage or group_leakage:
    raise RuntimeError("Cross-split overlap remains!")


print("\n===== CLASSIFIER V2 SUMMARY =====")

for split in ("train", "val", "test"):
    subset = [
        r for r in new_manifest if r["split"] == split
    ]

    counts = Counter(r["label"] for r in subset)

    print(
        f"{split}: {len(subset)} images | "
        f"tile={counts['tile']} | "
        f"not_tile={counts['not_tile']}"
    )

print("\nTotal images:", len(new_manifest))
print("Excluded images:", len(excluded_records))
print("Exact hash overlap:", exact_leakage)
print("Known source-group overlap:", group_leakage)
print("Output:", OUTPUT)
print("Exclusions report:", QUARANTINE_REPORT)
print("\nOriginal dataset v1 preserved.")

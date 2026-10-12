
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import random
import re
import shutil

ROOT = Path("data")
OUTPUT = ROOT / "processed/tile_classifier_v1"

DEFECT_DATA = ROOT / "processed/ceramic_defects_v1"
NORMAL_DATA = ROOT / "external/ceramic_surface_v4"
NEGATIVE_DATA = ROOT / "curated/non_tile_caltech101"

SPLITS = ("train", "val", "test")
ORIGINAL_SPLITS = ("train", "valid", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42

if OUTPUT.exists():
    raise SystemExit(
        f"Output already exists: {OUTPUT}\n"
        "Review it before running this script again."
    )

records = []


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group(path, origin, category=""):
    stem = path.stem.split(".rf.")[0]

    if origin == "caltech":
        return f"caltech:{category}:{stem}"

    mg = re.search(
        r"(?<![A-Za-z])MG_(\d+)(?:_|$)",
        stem, re.IGNORECASE
    )
    if mg:
        return f"camera:MG_{mg.group(1)}"

    q = re.search(
        r"(?<!\d)(\d+)-Q\d+-[A-Za-z]",
        stem, re.IGNORECASE
    )
    if q:
        return f"qitem:{q.group(1)}"

    return f"{origin}:{stem}"


def add_record(path, label, origin, split, category=""):
    records.append({
        "path": path,
        "label": label,
        "origin": origin,
        "original_split": split,
        "category": category,
        "source_group": source_group(path, origin, category),
        "sha": file_hash(path)
    })


# 1. Existing annotated tile images
for split in ORIGINAL_SPLITS:
    folder = DEFECT_DATA / split / "images"

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in EXTENSIONS:
            add_record(
                path, "tile", "ceramic_defects", split
            )


# 2. Normal-only ceramic surface images
for split in ORIGINAL_SPLITS:
    csv_path = NORMAL_DATA / split / "_classes.csv"

    with csv_path.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        normal = row.get("normal", "").strip() == "1"

        has_other_label = any(
            row.get(c, "").strip() == "1"
            for c in ("crack", "scratch", "stain", "Unlabeled")
        )

        if not normal or has_other_label:
            continue

        path = NORMAL_DATA / split / row["filename"].strip()

        if not path.is_file():
            raise FileNotFoundError(path)

        add_record(
            path, "tile", "ceramic_surface", split
        )


# 3. Caltech non-tile candidate images
for category_dir in sorted(NEGATIVE_DATA.iterdir()):
    if not category_dir.is_dir():
        continue

    for path in sorted(category_dir.iterdir()):
        if path.suffix.lower() in EXTENSIONS:
            add_record(
                path, "not_tile", "caltech",
                "unsplit", category_dir.name
            )


print("\n===== CANDIDATE DATA =====")
print("Total:", len(records))
print("Labels:", dict(Counter(r["label"] for r in records)))
print("Origins:", dict(Counter(r["origin"] for r in records)))


# Union-find: group both matching source IDs
# and exact duplicate image content.
parent = list(range(len(records)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(a, b):
    a, b = find(a), find(b)
    if a != b:
        parent[b] = a


by_source = {}
by_hash = {}

for i, record in enumerate(records):
    source = record["source_group"]
    sha = record["sha"]

    if source in by_source:
        union(i, by_source[source])
    else:
        by_source[source] = i

    if sha in by_hash:
        union(i, by_hash[sha])
    else:
        by_hash[sha] = i


groups = defaultdict(list)

for i, record in enumerate(records):
    groups[find(i)].append(i)


# A connected group must not have conflicting
# tile / non-tile labels.
for key, indices in groups.items():
    labels = {records[i]["label"] for i in indices}
    if len(labels) > 1:
        raise RuntimeError(
            f"Conflicting labels in source group {key}: {labels}"
        )


# Separate strata for class and dataset origin.
strata = defaultdict(list)

for key, indices in groups.items():
    labels = {records[i]["label"] for i in indices}
    origins = {records[i]["origin"] for i in indices}

    label = next(iter(labels))
    origin = (
        next(iter(origins))
        if len(origins) == 1 else "mixed"
    )

    strata[(label, origin)].append(key)


rng = random.Random(SEED)
assignments = {}

for stratum, group_keys in sorted(strata.items()):
    rng.shuffle(group_keys)
    group_keys.sort(
        key=lambda k: len(groups[k]), reverse=True
    )

    total = sum(len(groups[k]) for k in group_keys)

    targets = {
        split: total * RATIOS[split]
        for split in SPLITS
    }

    current = Counter()

    for key in group_keys:
        chosen = max(
            SPLITS,
            key=lambda s: (
                targets[s] - current[s]
            ) / max(targets[s], 1)
        )

        assignments[key] = chosen
        current[chosen] += len(groups[key])


# Remove exact duplicate files, but only AFTER
# grouping. This preserves duplicate relationships.
seen_hashes = set()
manifest = []
skipped_duplicates = 0

for key, indices in groups.items():
    split = assignments[key]

    for i in indices:
        record = records[i]

        if record["sha"] in seen_hashes:
            skipped_duplicates += 1
            continue

        seen_hashes.add(record["sha"])

        path = record["path"]
        label = record["label"]

        prefix = record["origin"]

        if record["category"]:
            prefix += "_" + record["category"]

        filename = f"{prefix}_{path.name}"

        destination = OUTPUT / split / label / filename

        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        shutil.copy2(path, destination)

        manifest.append({
            "destination": str(destination),
            "label": label,
            "split": split,
            "origin": record["origin"],
            "original_split": record["original_split"],
            "source_group": record["source_group"],
            "group_id": str(key),
            "sha256": record["sha"],
            "source_path": str(path)
        })


# Verify source groups are not split.
group_splits = defaultdict(set)

for row in manifest:
    group_splits[row["group_id"]].add(row["split"])

leakage = sum(
    len(splits) > 1
    for splits in group_splits.values()
)

if leakage:
    raise RuntimeError("Source-group split leakage detected!")


manifest_path = OUTPUT / "manifest.csv"

with manifest_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(manifest[0].keys())
    )
    writer.writeheader()
    writer.writerows(manifest)


print("\n===== FINAL DATASET =====")
print("Excluded exact duplicate files:", skipped_duplicates)
print("Source-group overlap across splits:", leakage)

for split in SPLITS:
    subset = [r for r in manifest if r["split"] == split]

    counts = Counter(r["label"] for r in subset)

    print(
        f"{split}: {len(subset)} images "
        f"| tile={counts['tile']} "
        f"| not_tile={counts['not_tile']}"
    )

print("\nOutput:", OUTPUT)
print("Manifest:", manifest_path)
print("\nPreparation completed.")
print("Original datasets remain unchanged.")

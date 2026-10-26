from pathlib import Path
from collections import defaultdict
import hashlib
import csv
import re

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- SETTINGS ----------------
DATASET_ROOT = Path(
    r"C:\Users\ansh2\Desktop\CNN-D\data\raw\plantvillage dataset"
)
SECTIONS = ["color", "grayscale", "segmented"]
PROJECT_ROOT = DATASET_ROOT.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}
# ------------------------------------------


def sha256_file(path):
    """Calculate a SHA-256 hash without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_source_key(image_path):
    """
    Build a likely source-image key from PlantVillage-style filenames.

    Many PlantVillage filenames have a UUID prefix followed by '___' and a
    source identifier, for example:
        random-uuid___RS_HL 6271.JPG

    The part after '___' is used to help keep corresponding color/grayscale/
    segmented representations in the same split. If there is no '___',
    the filename stem is used. This is a heuristic and should be reviewed.
    """
    stem = image_path.stem
    if "___" in stem:
        stem = stem.split("___", 1)[1]
    stem = re.sub(r"\s+", " ", stem.strip().lower())
    return stem


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def main():
    missing_sections = [
        section for section in SECTIONS
        if not (DATASET_ROOT / section).is_dir()
    ]
    if missing_sections:
        raise FileNotFoundError(
            "These required dataset section folders were not found: "
            + ", ".join(missing_sections)
            + f"\nExpected dataset root: {DATASET_ROOT}"
        )

    records = []
    print("Scanning all three sections and calculating file hashes...")

    for section in SECTIONS:
        section_root = DATASET_ROOT / section

        for class_dir in sorted(p for p in section_root.iterdir() if p.is_dir()):
            for image_path in sorted(class_dir.rglob("*")):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    continue

                try:
                    file_hash = sha256_file(image_path)
                except Exception as error:
                    print(f"WARNING: Could not hash {image_path}: {error}")
                    continue

                records.append({
                    "section": section,
                    "class_name": class_dir.name,
                    "image_path": str(image_path),
                    "file_name": image_path.name,
                    "sha256": file_hash,
                    "source_key": canonical_source_key(image_path),
                })

    if not records:
        raise RuntimeError("No images were found in the three dataset sections.")

    df = pd.DataFrame(records)

    # Identical bytes must not be assigned to different labels.
    labels_per_hash = df.groupby("sha256")["class_name"].nunique()
    conflicting_hashes = set(labels_per_hash[labels_per_hash > 1].index)

    if conflicting_hashes:
        conflict_df = df[df["sha256"].isin(conflicting_hashes)].copy()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        conflict_path = REPORTS_DIR / "duplicate_hash_label_conflicts.csv"
        conflict_df.to_csv(conflict_path, index=False, encoding="utf-8-sig")
        raise ValueError(
            "Some byte-identical images are assigned to different classes. "
            f"Review {conflict_path} before splitting."
        )

    # Group images by class + normalized source filename, so likely transformed
    # versions of the same original leaf are assigned to the same split.
    # Also union exact byte duplicates even if filenames differ.
    row_ids = list(df.index)
    union_find = UnionFind(row_ids)

    first_by_source = {}
    first_by_hash = {}

    for row_id, row in df.iterrows():
        source_group_key = (row["class_name"], row["source_key"])
        file_hash = row["sha256"]

        if source_group_key in first_by_source:
            union_find.union(row_id, first_by_source[source_group_key])
        else:
            first_by_source[source_group_key] = row_id

        if file_hash in first_by_hash:
            union_find.union(row_id, first_by_hash[file_hash])
        else:
            first_by_hash[file_hash] = row_id

    df["group_id"] = [union_find.find(row_id) for row_id in row_ids]

    # A group must have one class label. Stop if heuristic matching joined
    # different labels; this is safer than creating a misleading split.
    labels_per_group = df.groupby("group_id")["class_name"].nunique()
    bad_groups = set(labels_per_group[labels_per_group > 1].index)
    if bad_groups:
        bad_df = df[df["group_id"].isin(bad_groups)].copy()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        bad_path = REPORTS_DIR / "source_group_label_conflicts.csv"
        bad_df.to_csv(bad_path, index=False, encoding="utf-8-sig")
        raise ValueError(
            "Source grouping linked images from different class labels. "
            f"Review {bad_path}; no split files were created."
        )

    # Split unique groups (not individual image files) to prevent known
    # corresponding representations or exact duplicates crossing splits.
    groups = (
        df[["group_id", "class_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    groups_per_class = groups.groupby("class_name").size()
    if (groups_per_class < 3).any():
        low_classes = groups_per_class[groups_per_class < 3].to_dict()
        raise ValueError(
            "At least one class has fewer than 3 source groups, so a reliable "
            f"stratified train/validation/test split cannot be made: {low_classes}"
        )

    train_val, test = train_test_split(
        groups,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=groups["class_name"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=(0.15 / 0.85),
        random_state=RANDOM_SEED,
        stratify=train_val["class_name"],
    )

    group_to_split = {}
    for split_name, split_df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        for group_id in split_df["group_id"]:
            group_to_split[group_id] = split_name

    df["split"] = df["group_id"].map(group_to_split)
    df["target_image_size"] = "224x224"
    df["target_color_mode"] = "RGB (convert during image loading)"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "all_sections_split_manifest.csv"
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    # Combined split files and section-specific split files.
    for split_name in ["train", "validation", "test"]:
        df[df["split"] == split_name].to_csv(
            OUTPUT_DIR / f"{split_name}_all_sections.csv",
            index=False,
            encoding="utf-8-sig",
        )

    for section in SECTIONS:
        section_df = df[df["section"] == section]
        for split_name in ["train", "validation", "test"]:
            section_df[section_df["split"] == split_name].to_csv(
                OUTPUT_DIR / f"{section}_{split_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    class_counts = (
        df.groupby(["section", "split", "class_name"])
        .size()
        .reset_index(name="image_count")
    )
    class_counts.to_csv(
        OUTPUT_DIR / "split_class_counts_all_sections.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Leakage checks: no group IDs or exact hashes may occur in more than one split.
    group_sets = {
        name: set(df.loc[df["split"] == name, "group_id"])
        for name in ["train", "validation", "test"]
    }
    hash_sets = {
        name: set(df.loc[df["split"] == name, "sha256"])
        for name in ["train", "validation", "test"]
    }

    print("\n=== ALL-SECTIONS DATASET SPLIT SUMMARY ===")
    print(f"Total image files scanned: {len(df)}")
    print(f"Sections: {', '.join(SECTIONS)}")
    print(f"Classes found across sections: {df['class_name'].nunique()}")
    print(f"Source groups used for splitting: {groups['group_id'].nunique()}")

    print("\nImage file counts by section:")
    for section in SECTIONS:
        count = int((df["section"] == section).sum())
        print(f"  {section}: {count}")

    print("\nImage file counts by split (all sections combined):")
    for split_name in ["train", "validation", "test"]:
        count = int((df["split"] == split_name).sum())
        print(f"  {split_name}: {count} ({count / len(df) * 100:.2f}%)")

    print("\n=== LEAKAGE CHECK ===")
    split_names = ["train", "validation", "test"]
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            group_overlap = len(group_sets[a] & group_sets[b])
            hash_overlap = len(hash_sets[a] & hash_sets[b])
            print(
                f"{a} vs {b}: shared source groups={group_overlap}, "
                f"shared exact hashes={hash_overlap}"
            )

    print(f"\nManifest and split CSV files saved in: {OUTPUT_DIR}")
    print("Per-class counts saved to: split_class_counts_all_sections.csv")
    print("Original images were not copied, resized, moved, or deleted.")
    print(
        "Note: source matching uses the filename suffix after '___' when "
        "present. It is a useful heuristic, not proof that every transformed "
        "image pair has been matched."
    )


if __name__ == "__main__":
    main()

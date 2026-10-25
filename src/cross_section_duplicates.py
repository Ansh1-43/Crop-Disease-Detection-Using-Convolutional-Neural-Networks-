from pathlib import Path
from collections import defaultdict
import hashlib
import csv

DATASET_ROOT = Path(
    r"C:\Users\ansh2\Desktop\CNN-D\data\raw\plantvillage dataset"
)

SECTIONS = ["color", "grayscale", "segmented"]
PROJECT_ROOT = DATASET_ROOT.parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not DATASET_ROOT.exists():
        print("ERROR: Dataset root not found:")
        print(DATASET_ROOT)
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    hash_locations = defaultdict(list)
    total_images = 0

    for section in SECTIONS:
        section_root = DATASET_ROOT / section

        if not section_root.exists():
            print(f"WARNING: Missing section folder: {section_root}")
            continue

        for image_path in section_root.rglob("*"):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue

            total_images += 1

            try:
                file_hash = sha256_file(image_path)
            except Exception as error:
                print(f"Could not hash {image_path}: {error}")
                continue

            relative_path = image_path.relative_to(section_root)
            class_name = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "(unknown)"
            )

            hash_locations[file_hash].append({
                "section": section,
                "class_name": class_name,
                "image_path": str(image_path),
            })

    duplicate_rows = []
    duplicate_group = 0
    cross_section_groups = 0
    cross_class_groups = 0

    for file_hash, locations in hash_locations.items():
        if len(locations) < 2:
            continue

        duplicate_group += 1

        sections_found = {item["section"] for item in locations}
        classes_found = {item["class_name"] for item in locations}

        is_cross_section = len(sections_found) > 1
        is_cross_class = len(classes_found) > 1

        if is_cross_section:
            cross_section_groups += 1

        if is_cross_class:
            cross_class_groups += 1

        for item in locations:
            duplicate_rows.append({
                "sha256": file_hash,
                "duplicate_group": duplicate_group,
                "copies_in_group": len(locations),
                "sections_in_group": "; ".join(sorted(sections_found)),
                "classes_in_group": "; ".join(sorted(classes_found)),
                "cross_section_duplicate": is_cross_section,
                "cross_class_duplicate": is_cross_class,
                "section": item["section"],
                "class_name": item["class_name"],
                "image_path": item["image_path"],
            })

    report_path = REPORTS_DIR / "all_sections_exact_duplicates.csv"

    write_csv(
        report_path,
        [
            "sha256",
            "duplicate_group",
            "copies_in_group",
            "sections_in_group",
            "classes_in_group",
            "cross_section_duplicate",
            "cross_class_duplicate",
            "section",
            "class_name",
            "image_path",
        ],
        duplicate_rows,
    )

    print("\n=== CROSS-SECTION DUPLICATE SUMMARY ===")
    print(f"Images scanned across available sections: {total_images}")
    print(f"Exact duplicate groups across all sections: {duplicate_group}")
    print(f"Groups appearing in multiple sections: {cross_section_groups}")
    print(f"Groups appearing across different classes: {cross_class_groups}")
    print(f"Report saved to: {report_path}")
    print("No images were deleted or changed.")


if __name__ == "__main__":
    main()
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import csv
from PIL import Image

# Dataset root on your computer
DATASET_ROOT = Path(
    r"C:\Users\ansh2\Desktop\CNN-D\data\raw\plantvillage dataset"
)

# The three dataset sections to check
SECTIONS = ["color", "grayscale", "segmented"]

# Reports will be saved here
PROJECT_ROOT = DATASET_ROOT.parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def sha256_file(path):
    """Return the SHA-256 hash of a file, reading it in chunks."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_csv(path, fieldnames, rows):
    """Write report data to a CSV file."""
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_section(section):
    """Check image quality for one dataset section."""
    section_root = DATASET_ROOT / section

    if not section_root.exists():
        print(f"\n[{section}] SKIPPED: folder not found:")
        print(section_root)

        return {
            "section": section,
            "status": "MISSING_FOLDER",
            "image_count": 0,
            "class_count": 0,
            "unreadable_count": 0,
            "duplicate_groups": 0,
        }

    class_counts = Counter()
    unreadable_rows = []
    hashes_to_paths = defaultdict(list)
    total_images = 0

    # Find the class folders inside this section
    class_dirs = sorted(
        path for path in section_root.iterdir()
        if path.is_dir()
    )

    # Check each class folder
    for class_dir in class_dirs:

        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        class_counts[class_dir.name] += len(image_paths)

        for image_path in image_paths:
            total_images += 1

            # Check whether the image can be opened and decoded
            try:
                with Image.open(image_path) as image:
                    image.verify()

            except Exception as error:
                unreadable_rows.append({
                    "section": section,
                    "class_name": class_dir.name,
                    "image_path": str(image_path),
                    "error": str(error),
                })

            # Calculate file hash to identify exact duplicates
            try:
                file_hash = sha256_file(image_path)
                hashes_to_paths[file_hash].append(image_path)

            except Exception as error:
                print(f"Could not hash {image_path}: {error}")

    # Prepare class-count report
    class_rows = [
        {
            "section": section,
            "class_name": class_name,
            "image_count": count,
        }
        for class_name, count in sorted(class_counts.items())
    ]

    # Prepare exact-duplicate report
    duplicate_rows = []
    duplicate_group_number = 0

    for file_hash, paths in hashes_to_paths.items():

        if len(paths) > 1:
            duplicate_group_number += 1

            for path in paths:
                duplicate_rows.append({
                    "section": section,
                    "sha256": file_hash,
                    "duplicate_group": duplicate_group_number,
                    "image_path": str(path),
                    "copies_in_group": len(paths),
                })

    # Save the three reports for this section
    write_csv(
        REPORTS_DIR / f"{section}_class_counts.csv",
        ["section", "class_name", "image_count"],
        class_rows,
    )

    write_csv(
        REPORTS_DIR / f"{section}_unreadable_images.csv",
        ["section", "class_name", "image_path", "error"],
        unreadable_rows,
    )

    write_csv(
        REPORTS_DIR / f"{section}_exact_duplicate_images.csv",
        [
            "section",
            "sha256",
            "duplicate_group",
            "image_path",
            "copies_in_group",
        ],
        duplicate_rows,
    )

    # Display results for this section
    print(f"\n--- {section.upper()} ---")
    print(f"Images checked: {total_images}")
    print(f"Disease/class folders: {len(class_dirs)}")
    print(f"Unreadable images: {len(unreadable_rows)}")
    print(f"Exact duplicate groups: {duplicate_group_number}")
    print(f"Reports saved with prefix: {section}_")

    return {
        "section": section,
        "status": "OK",
        "image_count": total_images,
        "class_count": len(class_dirs),
        "unreadable_count": len(unreadable_rows),
        "duplicate_groups": duplicate_group_number,
    }


def main():
    # Check whether the dataset folder exists
    if not DATASET_ROOT.exists():
        print("ERROR: Dataset root was not found:")
        print(DATASET_ROOT)
        print("Check that the folder path is correct.")
        return

    # Create reports folder if it does not exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    # Check all three dataset sections
    for section in SECTIONS:
        summaries.append(check_section(section))

    # Save a combined summary report
    summary_path = REPORTS_DIR / "dataset_quality_summary.csv"

    write_csv(
        summary_path,
        [
            "section",
            "status",
            "image_count",
            "class_count",
            "unreadable_count",
            "duplicate_groups",
        ],
        summaries,
    )

    # Display summary in the terminal
    print("\n=== SUMMARY ===")

    for item in summaries:
        print(
            f"{item['section']}: "
            f"status={item['status']}, "
            f"images={item['image_count']}, "
            f"classes={item['class_count']}, "
            f"unreadable={item['unreadable_count']}, "
            f"exact_duplicate_groups={item['duplicate_groups']}"
        )

    print(f"\nAll reports saved in: {REPORTS_DIR}")
    print(f"Summary report: {summary_path}")
    print(
        "No images were deleted or changed. "
        "Review reports before taking action."
    )


if __name__ == "__main__":
    main()
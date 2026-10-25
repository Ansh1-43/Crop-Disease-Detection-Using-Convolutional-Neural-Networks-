
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: SET YOUR DATASET LOCATION
# ==========================================

dataset_root = Path(
    r"C:\Users\ansh2\Desktop\CNN-D\data\raw\plantvillage dataset"
)

sections = ["color", "grayscale", "segmented"]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Save charts in the project folder
output_dir = Path(__file__).resolve().parent.parent
output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# STEP 2: CHECK THE DATASET PATH
# ==========================================

print("=" * 55)
print("CROP DISEASE DATASET EXPLORATION")
print("=" * 55)
print("Dataset path:", dataset_root)
print("Dataset root exists:", dataset_root.is_dir())

if not dataset_root.is_dir():
    print("\nERROR: Dataset root folder was not found.")
    print("Check the path above and make sure it is correct.")
    raise SystemExit(1)

section_counts = {}
class_counts_by_section = {}

# ==========================================
# STEP 3: COUNT IMAGES IN EACH SECTION
# ==========================================

for section in sections:
    section_path = dataset_root / section

    print(f"\n{'-' * 55}")
    print("SECTION:", section.upper())
    print("Folder exists:", section_path.is_dir())

    if not section_path.is_dir():
        print("Missing section:", section)
        continue

    class_counts = {}
    section_total = 0

    # Find disease-class folders
    class_folders = sorted(
        item for item in section_path.iterdir()
        if item.is_dir()
    )

    print("Number of disease classes:", len(class_folders))

    for class_folder in class_folders:
        image_files = [
            file for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in image_extensions
        ]

        count = len(image_files)
        class_counts[class_folder.name] = count
        section_total += count

        print(f"{class_folder.name}: {count} images")

    section_counts[section] = section_total
    class_counts_by_section[section] = class_counts

    empty_classes = [
        name for name, count in class_counts.items()
        if count == 0
    ]

    print("Total images in section:", section_total)
    print("Empty classes:", len(empty_classes))

    if empty_classes:
        print("Empty class names:", empty_classes)

# ==========================================
# STEP 4: PRINT FINAL SUMMARY
# ==========================================

print(f"\n{'=' * 55}")
print("FINAL DATASET SUMMARY")
print("=" * 55)

for section, count in section_counts.items():
    print(f"{section}: {count} images")

total_images = sum(section_counts.values())
print("Total across available sections:", total_images)
print("Note: Different sections may contain versions")
print("of the same underlying leaf images.")
print("=" * 55)

# ==========================================
# STEP 5: CHART 1 - IMAGES PER SECTION
# ==========================================

if section_counts:
    plt.figure(figsize=(8, 5))
    plt.bar(
        list(section_counts.keys()),
        list(section_counts.values())
    )
    plt.title("Number of Images by Dataset Section")
    plt.xlabel("Dataset Section")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    chart_path = output_dir / "dataset_sections.png"
    plt.savefig(chart_path, dpi=150)
    print("\nChart saved:", chart_path)
    plt.show()
    plt.close()

# ==========================================
# STEP 6: CHART 2 - IMAGES PER DISEASE CLASS
# ==========================================

# Use color as the primary dataset for this chart.
color_counts = class_counts_by_section.get("color", {})

if color_counts:
    names = list(color_counts.keys())
    counts = list(color_counts.values())

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(names)), counts)
    plt.xticks(
        range(len(names)),
        names,
        rotation=90,
        fontsize=8
    )
    plt.title("Number of Images per Disease Class - Color")
    plt.xlabel("Disease Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    chart_path = output_dir / "disease_class_counts.png"
    plt.savefig(chart_path, dpi=150)
    print("Chart saved:", chart_path)
    plt.show()
    plt.close()
else:
    print("Color class counts unavailable; second chart skipped.")

print("\nEDA completed.")

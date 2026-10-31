
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "plantvillage dataset"
)

SECTIONS = [
    "color",
    "grayscale",
    "segmented"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. SCAN DATASET
# ==================================================

print("=" * 65)
print("CROP DISEASE DETECTION - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_ROOT}"
    )

class_records = []
section_records = []
format_records = []
dimension_records = []
sample_images = {}

for section in SECTIONS:

    section_path = DATASET_ROOT / section

    if not section_path.exists():
        print(f"\nWARNING: Missing section: {section}")
        continue

    print(f"\nScanning section: {section}")

    class_dirs = sorted(
        path for path in section_path.iterdir()
        if path.is_dir()
    )

    total_images = 0
    empty_classes = 0
    format_counter = Counter()

    for class_dir in class_dirs:

        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        image_count = len(image_paths)
        total_images += image_count

        if image_count == 0:
            empty_classes += 1

        class_records.append({
            "section": section,
            "class_name": class_dir.name,
            "image_count": image_count
        })

        for path in image_paths:
            format_counter[path.suffix.lower()] += 1

        # Keep one representative image from each class
        if image_paths:
            sample_images.setdefault(
                section,
                {}
            )[class_dir.name] = image_paths[
                len(image_paths) // 2
            ]

        # Inspect a small sample of image dimensions
        for path in image_paths[:10]:

            try:
                with Image.open(path) as img:

                    width, height = img.size

                    dimension_records.append({
                        "section": section,
                        "class_name": class_dir.name,
                        "width": width,
                        "height": height,
                        "format": img.format
                    })

            except (
                OSError,
                UnidentifiedImageError
            ):
                continue

    for extension, count in format_counter.items():

        format_records.append({
            "section": section,
            "extension": extension,
            "image_count": count
        })

    section_records.append({
        "section": section,
        "classes_found": len(class_dirs),
        "total_images": total_images,
        "empty_classes": empty_classes
    })

    print(f"Classes found: {len(class_dirs)}")
    print(f"Total images: {total_images}")
    print(f"Empty classes: {empty_classes}")


# ==================================================
# 3. CREATE DATAFRAMES AND SAVE CSV REPORTS
# ==================================================

if not class_records:
    raise RuntimeError(
        "No image classes were found. Check the dataset structure."
    )

class_df = pd.DataFrame(class_records)
section_df = pd.DataFrame(section_records)

format_df = pd.DataFrame(format_records)
dimension_df = pd.DataFrame(dimension_records)

class_df.to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False
)

section_df.to_csv(
    OUTPUT_DIR / "section_summary.csv",
    index=False
)

format_df.to_csv(
    OUTPUT_DIR / "image_format_distribution.csv",
    index=False
)

dimension_df.to_csv(
    OUTPUT_DIR / "image_dimensions_sample.csv",
    index=False
)

# ==================================================
# 4. CHART 1 - IMAGE TOTALS PER SECTION
# ==================================================

plt.figure(figsize=(9, 5))

plt.bar(
    section_df["section"],
    section_df["total_images"]
)

plt.title(
    "Total Images Across Dataset Sections",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Dataset Section")
plt.ylabel("Number of Images")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "section_image_totals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 5. CHART 2 - COLOR CLASS DISTRIBUTION
# ==================================================

color_df = class_df[
    class_df["section"] == "color"
].sort_values("image_count")

if not color_df.empty:

    plt.figure(figsize=(12, 13))

    plt.barh(
        color_df["class_name"],
        color_df["image_count"]
    )

    plt.title(
        "Image Distribution Across PlantVillage Classes",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Number of Images")
    plt.ylabel("Disease Classes")

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "color_class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# 6. CHART 3 - SECTION VS CLASS COMPARISON
# ==================================================

pivot_df = class_df.pivot_table(
    index="class_name",
    columns="section",
    values="image_count",
    aggfunc="sum",
    fill_value=0
)

pivot_df = pivot_df.reindex(
    columns=SECTIONS,
    fill_value=0
)

plt.figure(figsize=(9, 13))

plt.imshow(
    pivot_df.values,
    aspect="auto",
    cmap="Greens"
)

plt.colorbar(
    label="Number of Images"
)

plt.xticks(
    range(len(pivot_df.columns)),
    pivot_df.columns
)

plt.yticks(
    range(len(pivot_df.index)),
    pivot_df.index,
    fontsize=7
)

plt.title(
    "Class Distribution Across Image Sections",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Dataset Section")
plt.ylabel("Disease Class")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "section_class_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 7. CHART 4 - REPRESENTATIVE IMAGE GALLERY
# ==================================================

color_samples = sample_images.get("color", {})

if color_samples:

    columns = 4
    rows = (len(color_samples) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, rows * 3)
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for index, (class_name, image_path) in enumerate(
        sorted(color_samples.items())
    ):

        try:
            with Image.open(image_path) as img:

                axes[index].imshow(
                    img.convert("RGB")
                )

            readable_name = (
                class_name
                .replace("___", "\n")
                .replace("_", " ")
            )

            axes[index].set_title(
                readable_name,
                fontsize=7
            )

            axes[index].axis("off")

        except (
            OSError,
            UnidentifiedImageError
        ):
            continue

    fig.suptitle(
        "Representative PlantVillage Color Images",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "color_sample_gallery.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# 8. PRINT FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("EDA SCANNING COMPLETED")
print("=" * 65)

print("\nSECTION SUMMARY:")
print(section_df.to_string(index=False))

print("\nTOTAL IMAGES SCANNED:")

print(
    int(section_df["total_images"].sum())
)

print("\nCSV REPORTS AND CHARTS SAVED TO:")

print(OUTPUT_DIR)

print("\nGenerated files:")
for file_path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file_path.name)

print("\nNo dataset images were modified or deleted.")

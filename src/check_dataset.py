
from pathlib import Path

# Paste the actual path to the parent folder
# containing color, grayscale, and segmented.
dataset_root = Path(
    r"\Users\ansh2\Desktop\CNN-D\data\raw\plantvillage dataset"
)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not dataset_root.is_dir():
    print("ERROR: Dataset parent folder not found.")
else:
    total_all = 0

    for section in ["color", "grayscale", "segmented"]:
        section_path = dataset_root / section

        print(f"\n{'=' * 50}")
        print(f"SECTION: {section.upper()}")
        print("=" * 50)

        if not section_path.is_dir():
            print("Folder not found:", section_path)
            continue

        class_folders = sorted(
            folder for folder in section_path.iterdir()
            if folder.is_dir()
        )

        section_total = 0
        empty_classes = []

        print("Number of classes:", len(class_folders))

        for folder in class_folders:
            count = sum(
                1 for file in folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in image_extensions
            )

            print(f"{folder.name}: {count} images")
            section_total += count

            if count == 0:
                empty_classes.append(folder.name)

        print("Section total:", section_total)
        print("Empty classes:", len(empty_classes))

        total_all += section_total

    print(f"\n{'=' * 50}")
    print("TOTAL ACROSS ALL SECTIONS:", total_all)
    print("=" * 50)

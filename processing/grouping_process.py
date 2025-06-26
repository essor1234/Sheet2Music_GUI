import os
import shutil
import re
from pathlib import Path

def organize_grouped_images_by_type_fixed(source_dir, output_root, data_type="clefs", move=False):
    """
    Organize clef/staffLine images to:
    output_root/pdf_name/page_x/group_x/{clefs|staffLines}/filename.jpg
    Only from fClef and gClef folders.
    """
    source_dir = Path(source_dir)
    output_root = Path(output_root)
    valid_classes = {"fClef", "gClef", "clef"}

    image_files = [f for f in source_dir.rglob("*")
                   if f.is_file() and f.suffix.lower() in {'.jpg', '.png'}
                   and f.parent.name in valid_classes]

    if not image_files:
        print(f"⚠️ No valid {data_type} images found in {source_dir}")
        return

    print(f"🔍 Found {len(image_files)} {data_type} images to organize...")

    for idx, img_path in enumerate(image_files):
        fname = img_path.name
        try:
            # Traverse to extract pdf_name and page_x
            parts = img_path.parts
            page_name = next(p for p in parts if p.startswith("page_"))
            pdf_index = parts.index(page_name) - 1
            pdf_name = parts[pdf_index]

            # Group from filename (e.g., "something_group_3_clef.jpg")
            match = re.search(r'group_(\d+)', fname)
            if not match:
                print(f"⚠️ Skipping {fname}: No group number found")
                continue
            group_id = f"group_{match.group(1)}"
        except Exception as e:
            print(f"❌ Failed to parse metadata from {img_path}: {e}")
            continue

        target_dir = output_root / pdf_name / page_name / group_id / data_type
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / fname

        try:
            if move:
                shutil.move(str(img_path), str(dest_path))
                action = "Moved"
            else:
                shutil.copy2(str(img_path), str(dest_path))
                action = "Copied"
            print(f"[{idx+1}/{len(image_files)}] {action}: {fname} → {target_dir}")
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

    print(f"\n✅ {data_type} image organization complete.")


def organize_images_by_group(source_dir, output_dir, move=False):
    """
    Organize images into subfolders based on their group number extracted from filenames.

    Args:
        source_dir (str): Directory containing the images (e.g., 'testing14_staff_group_5_clef_0.jpg')
        output_dir (str): Base directory where group subfolders (e.g., 'group_5') will be created
        move (bool): If True, move files; if False, copy files (default: False)

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Regex to extract group number (e.g., 'group_5' -> 5)
    pattern = r'group_(\d+)'

    # Get list of image files
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"⚠️ No images found in {source_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    # Process each image
    for idx, filename in enumerate(image_files):
        source_path = os.path.join(source_dir, filename)

        # Extract group number from filename
        match = re.search(pattern, filename)
        if not match:
            print(f"⚠️ Skipping {filename}: No group number found")
            continue

        group_num = match.group(1)  # e.g., '5'
        group_folder = f"group_{group_num}"  # e.g., 'group_5'
        group_path = os.path.join(output_dir, group_folder)

        # Create group subfolder
        os.makedirs(group_path, exist_ok=True)

        # Destination path
        dest_path = os.path.join(group_path, filename)

        try:
            # Move or copy the file
            if move:
                shutil.move(source_path, dest_path)
                action = "Moved"
            else:
                shutil.copy2(source_path, dest_path)
                action = "Copied"
            print(f"[{idx + 1}/{len(image_files)}] {action}: {filename} -> {group_path}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {str(e)}")

    print("\n🎉 All images organized.")


def process_clef_grouping(fclef_path, gclef_path, target_path):
    organize_images_by_group(
        source_dir=fclef_path,
        output_dir=target_path,
        move=False
    )

    # Organize the cropped images by group
    organize_images_by_group(
        source_dir=gclef_path,
        output_dir=target_path,
        move=False
    )


def process_measure_grouping(path, target_path):
    organize_images_by_group(
        source_dir=path,
        output_dir=target_path,
        move=False
    )

def group_all_sep_images_fixed(clef_sep_path, staffLine_only_path, grouping_path):
    """
    Wrapper to group clef and staff line images into final structure.
    """
    print("📂 Organizing clef images...")
    organize_grouped_images_by_type_fixed(clef_sep_path, grouping_path, data_type="clefs", move=False)

    print("\n📂 Organizing staff line images...")
    organize_grouped_images_by_type_fixed(staffLine_only_path, grouping_path, data_type="staffLines", move=False)

    print("\n✅ All grouping completed.")
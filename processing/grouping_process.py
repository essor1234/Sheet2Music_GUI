import os
import shutil
import re


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
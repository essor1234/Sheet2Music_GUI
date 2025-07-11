# import os
# import shutil
# import re
# from pathlib import Path
# import cv2
#
# def organize_images_by_group(source_dir, output_dir, move=False):
#     """
#     Organize images into subfolders based on their group number extracted from filenames.
#
#     Args:
#         source_dir (str): Directory containing the images (e.g., 'testing14_staff_group_5_clef_0.jpg')
#         output_dir (str): Base directory where group subfolders (e.g., 'group_5') will be created
#         move (bool): If True, move files; if False, copy files (default: False)
#
#     Returns:
#         None
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Regex to extract group number (e.g., 'group_5' -> 5)
#     pattern = r'group_(\d+)'
#
#     # Get list of image files
#     image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#
#     if not image_files:
#         print(f"⚠️ No images found in {source_dir}")
#         return
#
#     print(f"Processing {len(image_files)} images...")
#
#     # Process each image
#     for idx, filename in enumerate(image_files):
#         source_path = os.path.join(source_dir, filename)
#
#         # Extract group number from filename
#         match = re.search(pattern, filename)
#         if not match:
#             print(f"⚠️ Skipping {filename}: No group number found")
#             continue
#
#         group_num = match.group(1)  # e.g., '5'
#         group_folder = f"group_{group_num}"  # e.g., 'group_5'
#         group_path = os.path.join(output_dir, group_folder)
#
#         # Create group subfolder
#         os.makedirs(group_path, exist_ok=True)
#
#         # Destination path
#         dest_path = os.path.join(group_path, filename)
#
#         try:
#             # Move or copy the file
#             if move:
#                 shutil.move(source_path, dest_path)
#                 action = "Moved"
#             else:
#                 shutil.copy2(source_path, dest_path)
#                 action = "Copied"
#             print(f"[{idx + 1}/{len(image_files)}] {action}: {filename} -> {group_path}")
#         except Exception as e:
#             print(f"⚠️ Error processing {filename}: {str(e)}")
#
#     print("\n🎉 All images organized.")
#
#
#
#
# def remove_small_images(base_folder, min_width=10, min_height=10, verbose=True):
#     """
#     Recursively removes images that are smaller than the given width and height.
#
#     Args:
#         base_folder (str): Root folder to scan (including subfolders).
#         min_width (int): Minimum allowed image width.
#         min_height (int): Minimum allowed image height.
#         verbose (bool): Whether to print details.
#
#     Returns:
#         List[str]: Paths to removed images.
#     """
#     removed_images = []
#
#     for root, _, files in os.walk(base_folder):
#         for fname in files:
#             if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
#                 path = os.path.join(root, fname)
#                 img = cv2.imread(path)
#
#                 if img is None:
#                     if verbose:
#                         print(f"⚠️ Skipping unreadable image: {path}")
#                     continue
#
#                 h, w = img.shape[:2]
#                 if w < min_width or h < min_height:
#                     removed_images.append(path)
#                     os.remove(path)
#                     if verbose:
#                         print(f"🗑️ Removed: {path} ({w}x{h})")
#
#     if verbose:
#         print(f"\n✅ Finished cleanup. Total removed: {len(removed_images)}")
#
#     return removed_images
#
#
# def clean_clef_crops(clef_root_folder, min_width=20, min_height=20, verbose=True):
#     """
#     Cleans all clef-classified crop folders under each page by removing small/invalid images.
#
#     Args:
#         clef_root_folder (str): Path to clef results root (e.g., clef_results/sheet/)
#         min_width (int): Minimum accepted width
#         min_height (int): Minimum accepted height
#         verbose (bool): Show messages
#
#     Returns:
#         List[str]: All removed image paths
#     """
#     removed = []
#
#     # Loop through each page folder (e.g., page_1, page_2, etc.)
#     page_dirs = sorted([
#         d for d in os.listdir(clef_root_folder)
#         if os.path.isdir(os.path.join(clef_root_folder, d)) and d.startswith("page_")
#     ])
#
#     for page in page_dirs:
#         page_path = os.path.join(clef_root_folder, page)
#
#         # Each label folder is a class like gClef, fClef, etc.
#         label_dirs = [
#             d for d in os.listdir(page_path)
#             if os.path.isdir(os.path.join(page_path, d))
#         ]
#
#         for label in label_dirs:
#             label_path = os.path.join(page_path, label)
#             if verbose:
#                 print(f"\n🔍 Cleaning: {label_path}")
#             removed += remove_small_images(label_path, min_width, min_height, verbose)
#
#     if verbose:
#         print(f"\n🧹 Done cleaning clef crops. Total removed: {len(removed)}")
#
#     return removed
#
#
#
#
#
# def remove_small_images(dir_path, min_width=20, min_height=20, verbose=False):
#     """
#     Remove small or unreadable images in a directory.
#     """
#     removed = []
#     for img_file in Path(dir_path).glob("*.[jp][pn]g"):
#         try:
#             img = cv2.imread(str(img_file))
#             if img is None or img.shape[1] < min_width or img.shape[0] < min_height:
#                 img_file.unlink()
#                 removed.append(str(img_file))
#                 if verbose:
#                     print(f"🗑️ Removed: {img_file.name} (invalid or too small)")
#         except Exception as e:
#             print(f"⚠️ Error reading {img_file}: {e}")
#             continue
#     return removed
#
#
# def clean_measure_crops(grouping_path, pdf_path, min_width=20, min_height=20, verbose=True):
#     """
#     Cleans all measure crops under:
#         grouping_path/pdf_name/page_x/group_x/measures/measure/
#
#     Args:
#         grouping_path (str or Path): Root directory (e.g., 'data_storage/grouping_path')
#         pdf_name (str): Name of the PDF folder to clean
#         min_width (int): Minimum allowed image width
#         min_height (int): Minimum allowed image height
#         verbose (bool): Whether to print cleaning progress
#
#     Returns:
#         List[str]: Paths of removed images
#     """
#     pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     pdf_dir = Path(grouping_path) / pdf_name
#     if not pdf_dir.exists():
#         print(f"❌ PDF directory does not exist: {pdf_dir}")
#         return []
#
#     removed_total = []
#
#     for page_dir in pdf_dir.glob("page_*"):
#         if not page_dir.is_dir():
#             continue
#         for group_dir in page_dir.glob("group_*"):
#             if not group_dir.is_dir():
#                 continue
#
#             measure_dir = group_dir / "measures" / "measure"
#             if not measure_dir.exists():
#                 if verbose:
#                     print(f"⛔ Skipping missing: {measure_dir}")
#                 continue
#
#             if verbose:
#                 print(f"\n🔍 Cleaning: {measure_dir}")
#
#             removed = remove_small_images(measure_dir, min_width, min_height, verbose)
#             removed_total.extend(removed)
#
#     if verbose:
#         print(f"\n✅ Measure cleaning complete. Total removed: {len(removed_total)}")
#
#     return removed_total
#
# ========================================================

import os
import shutil
import re
import cv2
from pathlib import Path


class ImageOrganizer:
    @staticmethod
    def organize_by_group(source_dir, output_dir, move=False):
        os.makedirs(output_dir, exist_ok=True)
        pattern = r'group_(\d+)'
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f" No images found in {source_dir}")
            return

        print(f"Processing {len(image_files)} images...")
        for idx, filename in enumerate(image_files):
            match = re.search(pattern, filename)
            if not match:
                print(f"Skipping {filename}: No group number found")
                continue

            group_folder = f"group_{match.group(1)}"
            group_path = os.path.join(output_dir, group_folder)
            os.makedirs(group_path, exist_ok=True)

            src = os.path.join(source_dir, filename)
            dst = os.path.join(group_path, filename)

            try:
                if move:
                    shutil.move(src, dst)
                    action = "Moved"
                else:
                    shutil.copy2(src, dst)
                    action = "Copied"
                print(f"[{idx + 1}/{len(image_files)}] {action}: {filename} -> {group_path}")
            except Exception as e:
                print(f" Error processing {filename}: {e}")
        print(" All images organized.")


class ImageCleaner:
    @staticmethod
    @staticmethod
    def remove_small_images(folder, min_width=20, min_height=20, verbose=False):
        """Removes images that are smaller than the specified dimensions."""
        removed = []
        for img_path in Path(folder).rglob("*.[jp][pn]g"):
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.shape[1] < min_width or img.shape[0] < min_height:
                    img_path.unlink()
                    removed.append(str(img_path))
                    if verbose:
                        print(f" Removed (too small): {img_path.name}")
            except Exception as e:
                print(f" ⚠️ Error reading {img_path}: {e}")
        return removed

    # --- NEW METHOD ---
    @staticmethod
    def remove_large_images(folder, max_height=200, verbose=False):
        """Removes images that are taller than the specified max_height."""
        removed = []
        for img_path in Path(folder).rglob("*.[jp][pn]g"):
            try:
                img = cv2.imread(str(img_path))
                # If the image's height is greater than the maximum allowed height
                if img is not None and img.shape[0] > max_height:
                    img_path.unlink()
                    removed.append(str(img_path))
                    if verbose:
                        print(f" Removed (too large): {img_path.name}")
            except Exception as e:
                print(f" ⚠️ Error reading {img_path}: {e}")
        return removed

    # --- UPDATED ORCHESTRATOR ---
    @staticmethod
    def clean_clef_crops(root_folder, min_width=20, min_height=20, max_height=140, verbose=True):
        """
        Cleans crop directories by removing both excessively small and large images.
        """
        removed_files = []
        for page_dir in Path(root_folder).glob("page_*"):
            if not page_dir.is_dir():
                continue

            # The label_dir is the folder containing the actual crops, e.g., 'clef' or 'measure'
            for label_dir in page_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                if verbose:
                    print(f"\n--- Cleaning: {label_dir} ---")

                # Run both cleaning steps
                removed_files += ImageCleaner.remove_small_images(label_dir, min_width, min_height, verbose)
                removed_files += ImageCleaner.remove_large_images(label_dir, max_height, verbose)

        if verbose:
            print("\n--------------------------")
            print(f"✅ Done cleaning. Total files removed: {len(removed_files)}")
        return removed_files


    @staticmethod
    def clean_measure_crops(grouping_path, pdf_path, min_width=20, min_height=20, verbose=True):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_dir = Path(grouping_path) / pdf_name
        if not pdf_dir.exists():
            print(f"PDF directory does not exist: {pdf_dir}")
            return []

        removed_total = []
        for page_dir in pdf_dir.glob("page_*"):
            for group_dir in page_dir.glob("group_*"):
                measure_dir = group_dir / "measures" / "measure"
                if not measure_dir.exists():
                    if verbose:
                        print(f"Skipping missing: {measure_dir}")
                    continue
                if verbose:
                    print(f"Cleaning: {measure_dir}")
                removed = ImageCleaner.remove_small_images(measure_dir, min_width, min_height, verbose)
                removed_total.extend(removed)
        if verbose:
            print(f"Measure cleaning complete. Total removed: {len(removed_total)}")
        return removed_total


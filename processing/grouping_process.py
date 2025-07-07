import os
import shutil
import re
from pathlib import Path
from typing import Dict


def organize_grouped_images_by_type_with_clef(
    staff_source_dir: str,
    clef_source_dir: str,
    output_root: str,
    move: bool = False,
):
    """
    Organize staff line images into structured output folders based on clef names.

    Each staff image is matched with a clef image by base name.
    The clef label (gClef or fClef) is extracted from the clef filename and appended to the staff filename.

    Output structure:
        output_root/pdf_name/page_x/group_x/staffLines/filename_gClef.jpg

    If no clef match is found, "_unknown" is added to the filename.
    """
    staff_source_dir = Path(staff_source_dir)
    clef_source_dir = Path(clef_source_dir)
    output_root = Path(output_root)

    print("📂 Organizing staff line images with clef labels...")

    # Step 1: Build clef mapping: base name → clef label
    clef_map: Dict[str, str] = {}
    for clef_img in clef_source_dir.rglob("*.*"):
        if clef_img.suffix.lower() not in {".jpg", ".png"}:
            continue
        if clef_img.parent.name not in {"fClef", "gClef"}:
            continue

        clef_label = clef_img.parent.name  # 'fClef' or 'gClef'
        clef_base = clef_img.stem
        # Remove the trailing _gClef_1 or _fClef_1 or similar
        match = re.match(r"(.+?)_(?:gClef|fClef)(?:_\d+)?$", clef_base)

        if match:
            staff_base = match.group(1)
        else:
            staff_base = clef_base
        clef_map[staff_base] = clef_label

    print(f"🔍 Found {len(clef_map)} clef-to-staff mappings.")

    # Step 2: Process staff images
    staff_images = [
        f for f in staff_source_dir.rglob("*.*")
        if f.suffix.lower() in {".jpg", ".png"}
    ]

    if not staff_images:
        print(f"⚠️ No staff line images found in {staff_source_dir}")
        return

    print(f"📦 Found {len(staff_images)} staff images to organize.")

    for idx, staff_path in enumerate(staff_images):
        try:
            fname = staff_path.name
            staff_base = staff_path.stem

            # Find page and pdf_name from path
            parts = staff_path.parts
            page_name = next(p for p in parts if p.startswith("page_"))
            pdf_index = parts.index(page_name) - 1
            pdf_name = parts[pdf_index]

            # Find group number
            match = re.search(r"group_(\d+)", fname)
            if not match:
                print(f"⚠️ Skipping {fname}: No group number found")
                continue
            group_id = f"group_{match.group(1)}"

            # Find clef label for this staff line
            clef_label = clef_map.get(staff_base, "unknown")
            new_name = f"{staff_base}_{clef_label}{staff_path.suffix}"

            # Output path
            target_dir = output_root / pdf_name / page_name / group_id / "staffLines"
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_path = target_dir / new_name

            if move:
                shutil.move(str(staff_path), str(dest_path))
                action = "Moved"
            else:
                shutil.copy2(staff_path, dest_path)
                action = "Copied"

            print(f"[{idx + 1}/{len(staff_images)}] {action}: {fname} → {new_name}")

        except Exception as e:
            print(f"❌ Error processing {staff_path}: {e}")

    print("\n✅ Staff line organization with clef names complete.")

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
    #
    print("\n📂 Organizing staff line images...")
    # organize_grouped_images_by_type_fixed(staffLine_only_path, grouping_path, data_type="staffLines", move=False)
    #
    # print("\n✅ All grouping completed.")
    organize_grouped_images_by_type_with_clef(
        staff_source_dir=staffLine_only_path,
        clef_source_dir=clef_sep_path,
        output_root=grouping_path,
        move=False
    )
#======================================GROUPING FOR PITCH RESULT==========================

import re
from collections import defaultdict


# def group_notes_by_page_group_measure_clef(nested_results):
#     structure = defaultdict(  # page
#         lambda: defaultdict(  # group
#             lambda: defaultdict(  # measure
#                 lambda: defaultdict(list)  # clef: list of notes
#             )
#         )
#     )
#
#     for group_key, group_data in nested_results.items():
#         for filename, notes in group_data.items():
#             # Example: twinkle-twinkle-little-star-piano-solo_page_1_staff_group_0_clef_1_gClef_measure_3.jpg
#             match = re.search(r'page_(\d+).*?group_(\d+).*?clef_\d+_(gClef|fClef)_measure_(\d+)', filename)
#             if not match:
#                 print(f"⚠️ Skipping: Filename not matched → {filename}")
#                 continue
#
#             page, group, clef, measure = match.groups()
#             page_key = f"page_{page}"
#             group_key = f"group_{group}"
#             measure_key = f"measure_{measure}"
#             clef_key = clef
#
#             structure[page_key][group_key][measure_key][clef_key].extend(notes)
#
#     return structure

import re
from collections import defaultdict

def group_notes_by_page_group_measure_clef(nested_results):
    structure = defaultdict(  # page
        lambda: defaultdict(  # group
            lambda: defaultdict(  # measure
                dict  # clef_index: { 'clef_type': ..., 'notes': [...] }
            )
        )
    )

    for group_key, group_data in nested_results.items():
        for filename, notes in group_data.items():
            # Example: ..._page_1_staff_group_0_clef_1_gClef_measure_3.jpg
            match = re.search(r'page_(\d+).*?group_(\d+).*?clef_(\d+)_(gClef|fClef)_measure_(\d+)', filename)
            if not match:
                print(f"⚠️ Skipping: Filename not matched → {filename}")
                continue

            page, group, clef_index, clef_type, measure = match.groups()
            page_key = f"page_{page}"
            group_key = f"group_{group}"
            measure_key = f"measure_{measure}"
            clef_index = int(clef_index)

            structure[page_key][group_key][measure_key][clef_index] = {
                "clef_type": clef_type,
                "notes": notes
            }

    # Optional: sort clefs by clef_index
    # This can be useful if you need ordered top-to-bottom clefs
    for page_dict in structure.values():
        for group_dict in page_dict.values():
            for measure_key, measure_dict in group_dict.items():
                group_dict[measure_key] = dict(sorted(measure_dict.items(), key=lambda x: x[0]))

    return structure






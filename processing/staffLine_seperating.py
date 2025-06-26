import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def separate_staffLine(img, filterVal, isDisplay=False):
    """
    Remove staff lines from a binary image using inpainting for better visual quality.

    Parameters:
        img: Input image (expected to be grayscale, 2D array or 3D single-channel array).
        filterVal: Threshold for detecting staff lines based on row mean intensity.
        isDisplay: If True, display original, staff-removed, and staff-only images.

    Returns:
        img_no_staff: Image with staff lines removed via inpainting.
        staff_lines: Image showing only the detected staff lines.
    """
    # Validate input image
    if img is None:
        raise ValueError("Input image is None")

    # Handle grayscale images (2D or 3D single-channel)
    if len(img.shape) == 3:
        if img.shape[2] != 1:
            raise ValueError(f"Expected single-channel grayscale image, got {img.shape[2]} channels")
        img = img.squeeze()  # Convert (height, width, 1) to (height, width)
    elif len(img.shape) != 2:
        raise ValueError(f"Expected grayscale image (2D array or 3D single-channel), got shape {img.shape}")

    # Binarize using Otsu's method
    _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create staff_lines image with same size as the binarized image
    staff_lines = np.full_like(binary_otsu, 255, dtype=np.uint8)

    # Detect staff lines based on row means
    row_means = np.mean(binary_otsu, axis=1)
    staff_rows = np.where(row_means < filterVal)[0]

    # Validate staff_rows against the image height
    if staff_rows.size > 0 and np.max(staff_rows) >= binary_otsu.shape[0]:
        print(
            f"Warning: Invalid staff_rows indices (max {np.max(staff_rows)} > {binary_otsu.shape[0] - 1}). Clipping to valid range.")
        staff_rows = staff_rows[staff_rows < binary_otsu.shape[0]]  # Clip invalid indices

    # Create a binary mask for staff lines (255 for staff pixels, 0 elsewhere)
    mask = np.zeros_like(binary_otsu, dtype=np.uint8)
    mask[staff_rows] = 255  # Mark staff rows in the mask

    # Refine the mask using morphological operations to isolate thin horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # Horizontal kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Inpaint the staff lines in the binarized image using the mask
    img_no_staff = cv2.inpaint(binary_otsu, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Update staff_lines to show only the detected staff lines
    staff_lines[mask == 255] = 0  # Black staff lines at detected positions

    if isDisplay:
        # Display the original binarized image
        print("Original (Binarized):")
        plt.imshow(binary_otsu, cmap='gray')
        plt.axis('off')
        plt.show()

        # Display the staff-removed image
        print("Staff Removed (Inpainted):")
        plt.imshow(img_no_staff, cmap='gray')
        plt.axis('off')
        plt.show()

        # Display the staff lines image
        print("Staff Lines Only:")
        plt.imshow(staff_lines, cmap='gray')
        plt.axis('off')
        plt.show()

    return img_no_staff, staff_lines


def process_groups_dataset(groups_path, output_noStaff_base_dir, output_staff_base_dir):
    """
    Process images in group subfolders to separate staff lines and save outputs in group-specific directories.

    Args:
        groups_path (str): Directory containing group subfolders (e.g., 'group_1', 'group_2')
        output_noStaff_base_dir (str): Base directory for staff-removed images (group subfolders will be created)
        output_staff_base_dir (str): Base directory for staff lines images (group subfolders will be created)

    Returns:
        None
    """
    # Create base output directories
    os.makedirs(output_noStaff_base_dir, exist_ok=True)
    os.makedirs(output_staff_base_dir, exist_ok=True)

    # Get list of group subfolders
    group_folders = [f for f in os.listdir(groups_path) if
                     os.path.isdir(os.path.join(groups_path, f)) and f.startswith('group_')]

    if not group_folders:
        print(f"⚠️ No group folders found in {groups_path}")
        return

    print(f"Found {len(group_folders)} group folders: {group_folders}")

    # Process each group folder
    for group_folder in group_folders:
        input_group_dir = os.path.join(groups_path, group_folder)
        output_noStaff_group_dir = os.path.join(output_noStaff_base_dir, group_folder)
        output_staff_group_dir = os.path.join(output_staff_base_dir, group_folder)

        # Create group-specific output directories
        os.makedirs(output_noStaff_group_dir, exist_ok=True)
        os.makedirs(output_staff_group_dir, exist_ok=True)

        print(f"\nProcessing group: {group_folder}")

        # Iterate through images in the group folder
        image_files = [f for f in os.listdir(input_group_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"⚠️ No images found in {input_group_dir}")
            continue

        for filename in image_files:
            img_path = os.path.join(input_group_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Failed to load {filename}, skipping...")
                continue

            # Debug: Print image shape
            print(f"Processing {filename} with shape {img.shape}")

            # Process the image
            try:
                noStaff, staff = separate_staffLine(img, filterVal=200, isDisplay=False)
            except Exception as e:
                print(f"Error processing {filename} in {group_folder}: {e}")
                continue

            # Save the processed images
            noStaff_path = os.path.join(output_noStaff_group_dir, filename)
            staff_path = os.path.join(output_staff_group_dir, filename)
            cv2.imwrite(noStaff_path, noStaff)
            cv2.imwrite(staff_path, staff)
            print(f"Processed and saved: {filename} -> {noStaff_path}, {staff_path}")

    print("\n🎉 All group folders processed.")

def separate_staff_from_clefs_flat(clef_input_root, output_noStaff_root, output_staff_root, filterVal=160):
    """
    Separates staff lines from clef crops in a flat `clef` subfolder under each page.

    Args:
        clef_input_root (str): Path from clefs_separating(), e.g., clef_results/sheet/
        output_noStaff_root (str): Where to save clef images with staff lines removed
        output_staff_root (str): Where to save extracted staff lines
        filterVal (int): Row threshold to detect staff lines

    Returns:
        Tuple[str, str]: Paths to (staff-removed images, staff-only images)
    """
    pdf_name = os.path.basename(clef_input_root.rstrip("/\\"))
    noStaff_pdf_root = os.path.join(output_noStaff_root, pdf_name)
    staff_pdf_root = os.path.join(output_staff_root, pdf_name)
    os.makedirs(noStaff_pdf_root, exist_ok=True)
    os.makedirs(staff_pdf_root, exist_ok=True)

    # Get page folders (e.g., page_1, page_2)
    page_dirs = sorted([
        d for d in os.listdir(clef_input_root)
        if os.path.isdir(os.path.join(clef_input_root, d)) and d.startswith("page_")
    ])

    for page_name in page_dirs:
        page_input_path = os.path.join(clef_input_root, page_name, "clef")
        if not os.path.isdir(page_input_path):
            print(f"⚠️ Skipping {page_name}: No 'clef' folder found.")
            continue

        output_noStaff_dir = os.path.join(noStaff_pdf_root, page_name, "clef")
        output_staff_dir = os.path.join(staff_pdf_root, page_name, "clef")
        os.makedirs(output_noStaff_dir, exist_ok=True)
        os.makedirs(output_staff_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(page_input_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        print(f"\n🧾 Processing {page_name}/clef ({len(image_files)} images)")

        for fname in image_files:
            img_path = os.path.join(page_input_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Failed to load {img_path}")
                continue

            try:
                no_staff, staff_only = separate_staffLine(img, filterVal=filterVal)

                noStaff_out = os.path.join(output_noStaff_dir, fname)
                staff_out = os.path.join(output_staff_dir, fname)

                cv2.imwrite(noStaff_out, no_staff)
                cv2.imwrite(staff_out, staff_only)
                print(f"✅ Saved: {noStaff_out}, {staff_out}")

            except Exception as e:
                print(f"⚠️ Error on {fname}: {e}")
                continue

    print("\n✅ Staff line removal complete.")
    return noStaff_pdf_root, staff_pdf_root

from itertools import groupby
from operator import itemgetter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def get_staffLine_y_coordinate(path, isDisplay=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Binarize (optional but recommended)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Compute mean value of each row
    row_mean = np.mean(binary, axis=1)  # 1 value per row

    # Threshold: stafflines are dark → after THRESH_BINARY_INV they become bright (high mean)
    threshold = np.max(row_mean) * 0.5  # adjust mul
    # Find row indices likely to contain staff lines
    staff_row_indices = np.where(row_mean > threshold)[0]
    if isDisplay:
        plt.plot(row_mean)
        plt.axhline(y=threshold, color='red', linestyle='--')
        plt.title("Row Mean Intensity (after binarization)")
        plt.xlabel("Row Index")
        plt.ylabel("Mean Pixel Intensity")
        plt.show()
    # print("Staff line row indices:", staff_row_indices)
    # Group consecutive rows (e.g., [120,121,122] → group as one line)
    groups = []
    for k, g in groupby(enumerate(staff_row_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        print(group)
        center = int(np.mean(group))
        groups.append(center)

    print("Estimated center rows of each staffline:", groups)

    return groups


def calculate_staffLine_distance(firstCoor, lastCorr):
    """
    firstCorr:int :y-coordinate of first line
    lastCorr:int  :y-coordinate of last line
    """
    return (lastCorr - firstCoor) / 4


# Calculate coordinate for staffLine and staffSpace
def calculate_staffs_y_coordinate(firstLineCoor, distance, staffType="line", extra=None):
    """
    firstLineCoor: Getting first staffLine y-coordinate
    distance: distance between 2 staffLine
    staffType: Calcualting mode for staffLine: "line" or staffSpace: "space"
    extra:int  : define the range of staff to calculate
    """
    # set start point
    if staffType == "line":
        start = 1
        end = 10
    elif staffType == "space":
        start = 0
        end = 12
    else:
        print("There's nothing called: ", staffType)
        return

    if extra:
        start -= extra * 2
        end += extra * 2

    staffList = []
    y1 = firstLineCoor
    d = distance
    for n in range(start, end, 2):
        y = int(y1 - d / 2 + n * d / 2)
        staffList.append((n, y))

    return staffList


def process_group_staffs(groups_path, output_folder, isDisplay=False, extra=None):
    """
    Process images in group folders to detect staff lines and calculate coordinates.

    Args:
        groups_path (str): Directory containing group subfolders (e.g., 'group_1', 'group_2').
        output_folder (str): Directory to save output files or results.
        isDisplay (bool): If True, display row mean intensity plots.

    Returns:
        dict: Mapping of group folder to list of staff coordinates per image.
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Get group folders
    group_folders = [f for f in os.listdir(groups_path) if
                     os.path.isdir(os.path.join(groups_path, f)) and f.startswith('group_')]

    if not group_folders:
        print(f"⚠️ No group folders found in {groups_path}")
        return {}

    print(f"Found {len(group_folders)} group folders: {group_folders}")

    results = {}  # Store results as {group_folder: {filename: staff_coordinates}}

    for group_folder in group_folders:
        input_group_dir = os.path.join(groups_path, group_folder)
        output_group_dir = os.path.join(output_folder, group_folder)
        os.makedirs(output_group_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_group_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"⚠️ No images found in {input_group_dir}")
            continue

        group_results = {}
        print(f"\nProcessing group: {group_folder} ({len(image_files)} images)")

        for filename in image_files:
            image_path = os.path.join(input_group_dir, filename)
            print(f"Processing: {filename}")

            # Get staff line y-coordinates
            staff_lines = get_staffLine_y_coordinate(image_path, isDisplay=isDisplay)

            if not staff_lines:
                print(f"⚠️ No staff lines detected in {filename}")
                continue

            # Calculate distance between staff lines (assuming 5 lines per staff)
            if len(staff_lines) >= 2:
                distance = calculate_staffLine_distance(staff_lines[0], staff_lines[-1])
            else:
                print(f"⚠️ Insufficient staff lines ({len(staff_lines)}) in {filename}, skipping distance calculation")
                continue

            # Calculate staff line and space coordinates
            staff_line_coords = calculate_staffs_y_coordinate(staff_lines[0], distance, staffType="line", extra=extra)
            staff_space_coords = calculate_staffs_y_coordinate(staff_lines[0], distance, staffType="space", extra=extra)

            # Store results
            group_results[filename] = {
                'staff_lines': staff_line_coords,
                'staff_spaces': staff_space_coords
            }

            # Optional: Save visualization or coordinates
            output_image_path = os.path.join(output_group_dir, f"{os.path.splitext(filename)[0]}_staff_lines.jpg")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for _, y in staff_line_coords:
                cv2.line(img, (0, y), (img.shape[1], y), (0), 1)
            cv2.imwrite(output_image_path, img)
            print(f"Saved visualization: {output_image_path}")

        results[group_folder] = group_results

    print("\n🎉 All group folders processed.")
    return results

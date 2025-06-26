from itertools import groupby
from operator import itemgetter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
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


def process_group_staffs_from_grouping(grouping_path, isDisplay=False, extra=None):
    """
    Process staff line detection inside grouping_path/.../group_x/staffLines/.

    Args:
        grouping_path (str or Path): Root path containing pdf/page/group/staffLines structure
        output_folder (str): Output path to save staffLine visualizations
        isDisplay (bool): Display row mean intensity plots (for debugging)
        extra (int): Extra lines/spaces before/after standard staff range

    Returns:
        dict: Nested dictionary with coordinates for each staff group
    """
    grouping_path = Path(grouping_path)
    # output_folder = Path(output_folder)
    # output_folder.mkdir(parents=True, exist_ok=True)

    results = {}

    for pdf_dir in grouping_path.iterdir():
        if not pdf_dir.is_dir():
            continue
        for page_dir in pdf_dir.glob("page_*"):
            for group_dir in page_dir.glob("group_*"):
                staff_dir = group_dir / "staffLines"
                if not staff_dir.exists():
                    print(f"⚠️ Skipping: No staffLines folder in {group_dir}")
                    continue

                image_files = [f for f in staff_dir.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                if not image_files:
                    print(f"⚠️ No images in {staff_dir}")
                    continue

                group_key = f"{pdf_dir.name}/{page_dir.name}/{group_dir.name}"
                results[group_key] = {}

                print(f"\n📄 Processing {group_key} ({len(image_files)} images)")

                for img_path in image_files:
                    print(f"Processing: {img_path.name}")
                    staff_lines = get_staffLine_y_coordinate(str(img_path), isDisplay=isDisplay)

                    if not staff_lines or len(staff_lines) < 2:
                        print(f"⚠️ Insufficient staff lines in {img_path.name}, skipping")
                        continue

                    distance = calculate_staffLine_distance(staff_lines[0], staff_lines[-1])
                    line_coords = calculate_staffs_y_coordinate(staff_lines[0], distance, "line", extra)
                    space_coords = calculate_staffs_y_coordinate(staff_lines[0], distance, "space", extra)

                    results[group_key][img_path.name] = {
                        "staff_lines": line_coords,
                        "staff_spaces": space_coords
                    }

                    # Optional visualization
                    # vis_dir = output_folder / pdf_dir.name / page_dir.name / group_dir.name
                    # vis_dir.mkdir(parents=True, exist_ok=True)
                    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    for _, y in line_coords:
                        cv2.line(img_gray, (0, y), (img_gray.shape[1], y), 0, 1)
                    # vis_out = vis_dir / f"{img_path.stem}_staff_lines.jpg"
                    # cv2.imwrite(str(vis_out), img_gray)
                    # print(f"📷 Saved: {vis_out.name}")

    print("\n✅ Finished processing all staffLines.")
    return results

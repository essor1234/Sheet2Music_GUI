from ultralytics import YOLO
import cv2
import numpy as np
import os
import random
from pathlib import Path
from torchvision.ops import nms
import torch

def resize_with_padding(img, target_size=(640, 640)):
    """Resize image to target_size while preserving aspect ratio and adding padding."""
    if img is None:
        print("⚠️ Skipping: image is None")
        return None, None

    if len(img.shape) == 2:  # Convert grayscale to RGB if needed
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        print(f"⚠️ Skipping: invalid image dimensions h={h}, w={w}")
        return None, None

    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h <= 0 or new_w <= 0:
        print(f"⚠️ Skipping: resized dimensions invalid new_h={new_h}, new_w={new_w}")
        return None, None

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    new_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    new_img[top:top + new_h, left:left + new_w] = resized
    return new_img, (left, top, new_h, new_w, scale)


def adjust_coords_to_original(coords, padding_info):
    """Adjust bounding box coordinates to original image scale."""
    left, top, _, _, scale = padding_info
    x_min, y_min, x_max, y_max = coords
    x_min = int((x_min - left) / scale)
    y_min = int((y_min - top) / scale)
    x_max = int((x_max - left) / scale)
    y_max = int((y_max - top) / scale)
    return [x_min, y_min, x_max, y_max]


def crop_grand_staff(img, coords):
    """Crop the image based on bounding box coordinates."""
    x_min, y_min, x_max, y_max = map(int, coords)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img.shape[1], x_max)
    y_max = min(img.shape[0], y_max)
    if x_max <= x_min or y_max <= y_min:
        print(f"Invalid crop coordinates: [{x_min}, {y_min}, {x_max}, {y_max}]")
        return None
    return img[y_min:y_max, x_min:x_max]


def draw_box(img, coords, label, conf, color=(0, 255, 0)):
    """Draw bounding box and label text."""
    x1, y1, x2, y2 = map(int, coords)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label_text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def crop_to_content(img, padding_info):
    """Crop out padding from image to return to original content area."""
    left, top, new_h, new_w, _ = padding_info
    return img[top:top + new_h, left:left + new_w]


def loading_YOLO_model_v2(model_path, input_source, output_folder=None, conf=0.75, iou=0.7, max_images=None,
                          save_crops=False, sort_by_x=False, sort_by_y=False, maximize_top_bottom=False, maximize_left_right=False):
    """Run YOLO model on a folder of images, a single image file, or a direct image array with annotation visualization."""

    # Load model
    global final_img
    try:
        model = YOLO(model_path)
        print("Loaded model with classes:", model.names)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model at {model_path}: {str(e)}")

    # Generate color map
    class_colors = {cid: tuple(random.randint(0, 255) for _ in range(3)) for cid in model.names}

    # Initialize variables
    image_files = []
    input_folder = None
    single_image = None
    is_direct_image = False

    # Determine input type
    if isinstance(input_source, np.ndarray):  # Direct image array
        is_direct_image = True
        single_image = input_source
        image_files = ["direct_image.jpg"]  # Dummy filename for processing
        if output_folder is None:
            raise ValueError("output_folder must be specified when using a direct image input")
        os.makedirs(output_folder, exist_ok=True)
    elif os.path.isfile(input_source):  # Single image file
        image_files = [os.path.basename(input_source)]
        input_folder = os.path.dirname(input_source) or "."
        if output_folder is None:
            output_folder = input_folder
        os.makedirs(output_folder, exist_ok=True)
    elif os.path.isdir(input_source):  # Directory of images
        input_folder = input_source
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if output_folder is None:
            output_folder = os.path.join(input_folder, "output")
        os.makedirs(output_folder, exist_ok=True)
    else:
        raise ValueError(f"Invalid input_source: {input_source}. Must be an image array, image file, or directory.")

    if max_images and not is_direct_image:
        image_files = image_files[:min(max_images, len(image_files))]
    print(f"Processing {len(image_files)} images...")

    # Process each image
    for idx, image_file in enumerate(image_files):
        if is_direct_image:
            print(f"\n[{idx + 1}/{len(image_files)}] Processing: Direct image input")
            img = single_image
            base_name = "direct_image"
        else:
            image_path = os.path.join(input_folder, image_file)
            print(f"\n[{idx + 1}/{len(image_files)}] Processing: {image_path}")
            img = cv2.imread(image_path)
            base_name = os.path.splitext(image_file)[0]
            if img is None:
                print(f"⚠️ Skipping: Could not load {image_path}")
                continue

        img_resized, padding_info = resize_with_padding(img, target_size=(640, 640))
        if img_resized is None or padding_info is None:
            print(f"⚠️ Skipping due to invalid resize: {image_file}")
            continue

        # results = model.predict(source=img_resized, conf=conf, iou=iou, imgsz=640)
        results = model.predict(source=img_resized, conf=conf, iou=iou, imgsz=640, nms=False)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_tensor = torch.tensor([box.xyxy[0].tolist() for box in result.boxes])
                scores_tensor = torch.tensor([box.conf.item() for box in result.boxes])
                keep = nms(boxes_tensor, scores_tensor, iou_threshold=iou)
                result.boxes = [result.boxes[i] for i in keep]  # 🔥 Apply filtered boxes back

        annotated_img = img_resized.copy()
        boxes_detected = False

        if results and len(results) > 0:
            for result in results:
                boxes = getattr(result, 'boxes', None)
                if boxes is not None and len(boxes) > 0:
                    boxes_detected = True

                    # Sort boxes by x_min (left to right)
                    if sort_by_x:
                        boxes = sorted(boxes, key=lambda b: b.xyxy[0][0].item())

                    if sort_by_y:
                        boxes = sorted(boxes, key=lambda b: b.xyxy[0][1].item())

                    for i, box in enumerate(boxes):
                        class_id = int(box.cls)
                        label = model.names[class_id]
                        confidence = box.conf.item()
                        coords = box.xyxy[0].tolist()

                        if save_crops:
                            # Adjust coordinates to remove padding and map to original image
                            x_min, y_min, x_max, y_max = adjust_coords_to_original(coords, padding_info)
                            if maximize_top_bottom:
                                #  Override top and bottom to maximize height
                                y_min = 0
                                y_max = img.shape[0]

                                # Make sure x is still within bounds
                                x_min = max(0, x_min)
                                x_max = min(img.shape[1], x_max)
                            elif maximize_left_right:
                                # Override left and right to maximize width
                                x_min = 0
                                x_max = img.shape[1]

                                # y still with bounds
                                y_min = max(0, y_min)
                                y_max = min(img.shape[0], y_max)
                            else:
                                x_min = max(0, x_min)
                                y_min = max(0, y_min)
                                x_max = min(img.shape[1], x_max)
                                y_max = min(img.shape[0], y_max)


                            # Crop from original image
                            crop = img[y_min:y_max, x_min:x_max]
                            if crop.size == 0:
                                print(f"⚠️ Empty crop for {label}, skipping.")
                                continue

                            # Save to: output_folder/class_name/image_classname_index.jpg
                            class_dir = os.path.join(output_folder, label)
                            os.makedirs(class_dir, exist_ok=True)
                            crop_filename = f"{base_name}_{label}_{i}.jpg"
                            crop_path = os.path.join(class_dir, crop_filename)
                            cv2.imwrite(crop_path, crop)
                            print(f"📦 Saved crop: {crop_path}")

                        color = class_colors[class_id]
                        print(f"🔍 {label} ({confidence:.2f}) at {coords}")
                        annotated_img = draw_box(annotated_img, coords, label, confidence, color=color)

        final_img = crop_to_content(annotated_img, padding_info)

        # Save result
        output_path = os.path.join(output_folder, f"{base_name}_detected.jpg")
        cv2.imwrite(output_path, final_img)
        print(f"✅ Saved to {output_path}")

    print("\n🎉 All images processed.")
    # Return the final annotated image for direct image input
    if is_direct_image:
        return final_img


def grandStaff_separating(model_path, binarized_folder, pdf_path, output_root):
    """
    Runs YOLO predictions on all binarized images and saves crops into:
    {output_root}/{pdf_name}/page_{n}/{label}/...

    Returns:
        str: The full path to the folder containing all grand staff crops.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    target_root = os.path.join(output_root, pdf_name)
    os.makedirs(target_root, exist_ok=True)

    images = sorted([
        f for f in os.listdir(binarized_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    for i, img_file in enumerate(images):
        img_path = os.path.join(binarized_folder, img_file)
        page_folder = os.path.join(target_root, f"page_{i + 1}")
        os.makedirs(page_folder, exist_ok=True)

        print(f"\n🔎 Predicting page {i+1}: {img_file}")
        loading_YOLO_model_v2(
            model_path=model_path,
            input_source=img_path,
            output_folder=page_folder,
            save_crops=True,
            sort_by_y=True
        )

    return target_root


def clefs_separating(model_path, grandstaff_folder, output_root):
    """
    Applies clef detection on images inside each page's 'staff_group' folder.

    Args:
        model_path (str): Path to the clef YOLO model.
        grandstaff_folder (str): Path from `predict_and_organize_by_page()`, e.g., clef_crops/sheet/
        output_root (str): Base folder to store clef results.

    Returns:
        str: Path to clef results organized by page.
    """
    pdf_name = os.path.basename(grandstaff_folder.rstrip("/\\"))
    output_pdf_root = os.path.join(output_root, pdf_name)
    os.makedirs(output_pdf_root, exist_ok=True)

    page_dirs = sorted([
        d for d in os.listdir(grandstaff_folder)
        if os.path.isdir(os.path.join(grandstaff_folder, d)) and d.startswith("page_")
    ])

    for page_name in page_dirs:
        page_path = os.path.join(grandstaff_folder, page_name)
        staff_group_path = os.path.join(page_path, "staff_group")

        if not os.path.isdir(staff_group_path):
            print(f"⚠️ Skipping {page_name}: 'staff_group' folder not found.")
            continue

        output_page_path = os.path.join(output_pdf_root, page_name)
        os.makedirs(output_page_path, exist_ok=True)

        print(f"\n🎼 Processing clefs in: {staff_group_path}")
        loading_YOLO_model_v2(
            model_path=model_path,
            input_source=staff_group_path,
            output_folder=output_page_path,
            save_crops=True,
            sort_by_y=True,
            conf=0.52,
            iou=0.2,
            maximize_left_right=True
        )

    return output_pdf_root


def measures_separating_from_grouping(model_path, grouping_path, pdf_path):
    """
    Detects and crops measures from clefs for the specified PDF under grouping_path.

    Input:  grouping_path/pdf_name/page_x/group_x/clefs/
    Output: grouping_path/pdf_name/page_x/group_x/measures/

    Args:
        model_path (str): Path to the YOLO model for measure detection.
        grouping_path (str or Path): Root directory containing PDF folders.
        pdf_name (str): Name of the PDF (used as folder name).
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_dir = Path(grouping_path) / pdf_name
    if not pdf_dir.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        return

    for page_dir in pdf_dir.glob("page_*"):
        if not page_dir.is_dir():
            continue

        for group_dir in page_dir.glob("group_*"):
            clef_input = group_dir / "clefs"
            if not clef_input.exists():
                print(f"⚠️ Skipping: No 'clefs/' in {group_dir}")
                continue

            measure_output = group_dir / "measures"
            measure_output.mkdir(parents=True, exist_ok=True)

            print(f"📍 Detecting measures: {clef_input} → {measure_output}")

            loading_YOLO_model_v2(
                model_path=model_path,
                input_source=str(clef_input),
                output_folder=str(measure_output),
                save_crops=True,
                conf=0.52,
                iou=0.1,
                sort_by_x=True,
                maximize_top_bottom=True
            )




import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import transforms
from PIL import Image
import cv2
import os
import numpy as np
import re
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_image_for_noteheads(img_np, enhance=False, blur_type='gaussian', blur_strength=3):
    """
    Preprocess image to enhance noteheads and reduce noise resembling noteheads.

    Args:
        img_np (numpy.ndarray): Input image in OpenCV format (BGR).
        enhance (bool): If True, apply contrast and morphological operations to enhance noteheads.
        blur_type (str): Type of blur to apply ('gaussian', 'median', 'bilateral', or None).
        blur_strength (int): Kernel size for blur (must be odd for Gaussian/median, diameter for bilateral).

    Returns:
        numpy.ndarray: Processed image.
    """
    img_processed = img_np.copy()

    if enhance:
        # Convert to grayscale for preprocessing
        gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        if blur_type == 'gaussian' and blur_strength > 0:
            # Ensure blur_strength is odd
            blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            gray = cv2.GaussianBlur(gray, (blur_strength, blur_strength), 0)
        elif blur_type == 'median' and blur_strength > 0:
            # Ensure blur_strength is odd
            blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            gray = cv2.medianBlur(gray, blur_strength)
        elif blur_type == 'bilateral' and blur_strength > 0:
            gray = cv2.bilateralFilter(gray, blur_strength, 75, 75)

        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Apply morphological closing to emphasize filled noteheads
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Convert back to RGB
        img_processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return img_processed


# def load_model(model_path, num_classes=5):
#     """
#     Load the Faster R-CNN model from the specified path.
#
#     Args:
#         model_path (str): Path to the trained model (.pth file).
#         num_classes (int): Number of classes (default: 5).
#
#     Returns:
#         model: Loaded Faster R-CNN model in evaluation mode.
#     """
#     model = fasterrcnn_resnet50_fpn_v2(pretrained=False, num_classes=num_classes)
#     state_dict = torch.load(model_path, map_location=device)
#     # Rename mismatched keys if needed
#     state_dict['roi_heads.box_predictor.cls_score.weight'] = state_dict.pop(
#         'roi_heads.box_predictor.cls_score.weight',
#         state_dict['roi_heads.box_predictor.cls_score.weight']
#     )
#     state_dict['roi_heads.box_predictor.cls_score.bias'] = state_dict.pop(
#         'roi_heads.box_predictor.cls_score.bias',
#         state_dict['roi_heads.box_predictor.cls_score.bias']
#     )
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model


def load_model(model_path, model_type='fasterrcnn_resnet50_fpn_v2', num_classes=5, pretrained_backbone=False):
    """
    Load a Faster R-CNN model of the specified type from the given path.

    Args:
        model_path (str): Path to the trained model (.pth file).
        model_type (str): Type of Faster R-CNN model (e.g., 'fasterrcnn_resnet50_fpn_v2',
                          'fasterrcnn_mobilenet_v3_large_fpn'). Must match a model in torchvision.models.detection.
        num_classes (int): Number of classes (default: 5).
        pretrained_backbone (bool): If True, use pretrained backbone weights (default: False).

    Returns:
        model: Loaded Faster R-CNN model in evaluation mode.

    Raises:
        ValueError: If model_type is not supported or invalid.
        RuntimeError: If model loading or state dict assignment fails.
    """
    # Validate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the model constructor from torchvision.models.detection
    try:
        import torchvision.models.detection as detection
        model_constructor = getattr(detection, model_type, None)
        if model_constructor is None:
            raise ValueError(
                f"Invalid model_type: {model_type}. Must be a valid model in torchvision.models.detection "
                "(e.g., 'fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_fpn')."
            )
    except ImportError:
        raise ImportError("torchvision.models.detection module not found. Ensure torchvision is installed.")

    # Initialize the model
    try:
        # Some models require different kwargs (e.g., pretrained vs pretrained_backbone)
        model = model_constructor(
            pretrained=pretrained_backbone if 'pretrained' in model_constructor.__code__.co_varnames else False,
            pretrained_backbone=pretrained_backbone,
            num_classes=num_classes
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model {model_type}: {str(e)}")

    # Load state dictionary
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {str(e)}")

    # Handle potential key mismatches in state dictionary
    model_state_dict = model.state_dict()
    updated_state_dict = {}
    for key, value in state_dict.items():
        # Try to match keys, accounting for common mismatches
        new_key = key
        if key not in model_state_dict:
            # Example: Handle roi_heads.box_predictor.cls_score.weight mismatch
            if 'roi_heads.box_predictor.cls_score.weight' in key:
                new_key = 'roi_heads.box_predictor.cls_score.weight'
            elif 'roi_heads.box_predictor.cls_score.bias' in key:
                new_key = 'roi_heads.box_predictor.cls_score.bias'
            # Add more key mappings if needed for other models
            else:
                print(f"⚠️ Skipping unmatched key in state dict: {key}")
                continue
        if model_state_dict[new_key].shape != value.shape:
            print(f"⚠️ Shape mismatch for key {new_key}: expected {model_state_dict[new_key].shape}, got {value.shape}")
            continue
        updated_state_dict[new_key] = value

    # Update model with compatible weights
    model_state_dict.update(updated_state_dict)
    try:
        model.load_state_dict(model_state_dict, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict into model: {str(e)}")

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    print(f"✅ Loaded model {model_type} with {num_classes} classes from {model_path} on {device}")
    return model

def load_and_preprocess_image(image_path, enhance_noteheads=False, blur_type='gaussian', blur_strength=3):
    """
    Load an image and optionally preprocess it to enhance noteheads.

    Args:
        image_path (str): Path to the input image.
        enhance_noteheads (bool): If True, apply preprocessing to enhance noteheads.
        blur_type (str): Type of blur ('gaussian', 'median', 'bilateral', or None).
        blur_strength (int): Kernel size for blur.

    Returns:
        tuple: (img_np, original_height, original_width)
            - img_np: Preprocessed image as NumPy array.
            - original_height, original_width: Original image dimensions.
    """
    img_np = cv2.imread(image_path)
    if img_np is None:
        print(f"⚠️ Failed to load {image_path}")
        return None, None, None
    original_height, original_width = img_np.shape[:2]
    if enhance_noteheads:
        img_np = preprocess_image_for_noteheads(img_np, enhance=True, blur_type=blur_type, blur_strength=blur_strength)
    return img_np, original_height, original_width

def resize_image(img_np, input_resize_factor, original_width, original_height):
    """
    Resize the input image by the specified factor.

    Args:
        img_np (np.ndarray): Input image as NumPy array.
        input_resize_factor (float): Factor to resize the image.
        original_width (int): Original image width.
        original_height (int): Original image height.

    Returns:
        tuple: (resized_img_np, new_width, new_height)
            - resized_img_np: Resized image as NumPy array.
            - new_width, new_height: Resized image dimensions.
    """
    if input_resize_factor == 1.0:
        return img_np, original_width, original_height
    new_width = int(original_width * input_resize_factor)
    new_height = int(original_height * input_resize_factor)
    resized_img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_img_np, new_width, new_height

def visualize_resized_image(img_np, filename, new_width, new_height, visualize_resized, visualize_dir):
    """
    Visualize or save the resized image.

    Args:
        img_np (np.ndarray): Resized image as NumPy array.
        filename (str): Image filename.
        new_width (int): Resized image width.
        new_height (int): Resized image height.
        visualize_resized (bool or str): 'display' to show, 'save' to save, False to skip.
        visualize_dir (str): Directory to save resized images if visualize_resized='save'.
    """
    if not visualize_resized:
        return
    if visualize_resized == 'display':
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        plt.title(f"Resized: {filename} ({new_width}x{new_height})")
        plt.axis('off')
        plt.show()
    elif visualize_resized == 'save' and visualize_dir:
        vis_path = os.path.join(visualize_dir, f"{os.path.splitext(filename)[0]}_resized.jpg")
        cv2.imwrite(vis_path, img_np)
        print(f"📸 Saved resized image: {vis_path} (size: {new_width}x{new_height})")

def run_inference(model, img_np):
    """
    Run model inference on the input image.

    Args:
        model: Loaded Faster R-CNN model.
        img_np (np.ndarray): Input image as NumPy array.

    Returns:
        dict: Model predictions (boxes, scores, labels).
    """
    image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return predictions

def process_predictions(predictions, conf_threshold, input_resize_factor):
    """
    Filter and normalize predictions, sorting by x1 coordinate.

    Args:
        predictions (dict): Model predictions (boxes, scores, labels).
        conf_threshold (float): Confidence threshold for predictions.
        input_resize_factor (float): Factor used to resize the input image.

    Returns:
        list: Filtered predictions, each with normalized bbox, label, and score, sorted by x1.
    """
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    filtered_predictions = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= conf_threshold:
            x1, y1, x2, y2 = box
            # Normalize bounding box to original image size
            x1_norm = x1 / input_resize_factor
            y1_norm = y1 / input_resize_factor
            x2_norm = x2 / input_resize_factor
            y2_norm = y2 / input_resize_factor
            filtered_predictions.append({
                'bbox': [float(x1_norm), float(y1_norm), float(x2_norm), float(y2_norm)],
                'label': int(label),
                'score': float(score)
            })
    return sorted(filtered_predictions, key=lambda x: x['bbox'][0])

def save_crops(img_np_original, filename, filtered_predictions, output_group_dir):
    """
    Save cropped bounding boxes from the original image.

    Args:
        img_np_original (np.ndarray): Original image as NumPy array.
        filename (str): Image filename.
        filtered_predictions (list): Filtered predictions with normalized bboxes.
        output_group_dir (str): Directory to save cropped images.
    """
    for i, pred in enumerate(filtered_predictions):
        x1, y1, x2, y2 = map(int, pred['bbox'])
        label = pred['label']
        crop = img_np_original[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"⚠️ Empty crop for {filename} at index {i}, skipping")
            continue
        crop_filename = f"{os.path.splitext(filename)[0]}_note_{i}_label_{label}.jpg"
        crop_path = os.path.join(output_group_dir, crop_filename)
        cv2.imwrite(crop_path, crop)
        print(f"📦 Saved crop: {crop_path} (size: {crop.shape[1]}x{crop.shape[0]})")

def save_image_with_bboxes(img_np, filename, filtered_predictions, input_resize_factor, bbox_images_dir, new_width, new_height):
    """
    Save the preprocessed image with bounding boxes (no labels).

    Args:
        img_np (np.ndarray): Preprocessed image as NumPy array.
        filename (str): Image filename.
        filtered_predictions (list): Filtered predictions with normalized bboxes.
        input_resize_factor (float): Factor used to resize the input image.
        bbox_images_dir (str): Directory to save images with bboxes.
        new_width (int): Resized image width.
        new_height (int): Resized image height.
    """
    img_np_with_boxes = img_np.copy()
    for pred in filtered_predictions:
        x1 = int(pred['bbox'][0] * input_resize_factor)
        y1 = int(pred['bbox'][1] * input_resize_factor)
        x2 = int(pred['bbox'][2] * input_resize_factor)
        y2 = int(pred['bbox'][3] * input_resize_factor)
        cv2.rectangle(img_np_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    bbox_path = os.path.join(bbox_images_dir, f"{os.path.splitext(filename)[0]}_bbox.jpg")
    cv2.imwrite(bbox_path, img_np_with_boxes)
    print(f"📸 Saved image with bounding boxes: {bbox_path} (size: {new_width}x{new_height})")

def predict_notes(model_path, groups_path, model_type='fasterrcnn_resnet50_fpn_v2', output_base_dir=None,
                  conf_threshold=0.5, num_classes=5, save_crops_enabled=False,  # Renamed parameter
                  input_resize_factor=2.0, enhance_noteheads=False, blur_type='gaussian', blur_strength=3,
                  visualize_resized=False, visualize_dir=None, save_bbox_images=False, bbox_images_dir=None):
    """
    Predict bounding boxes for notes in images across group folders, return sorted coordinates,
    and optionally save cropped bounding boxes or preprocessed images with bounding boxes (no labels).

    Args:
        model_path (str): Path to the trained Faster R-CNN model (.pth file).
        groups_path (str): Directory containing group subfolders (e.g., 'group_1', 'group_2').
        model_type (str, optional): Type of Faster R-CNN model (default: 'fasterrcnn_resnet50_fpn_v2').
        output_base_dir (str, optional): Base directory to save cropped bounding boxes.
        conf_threshold (float): Confidence threshold for predictions (default: 0.5).
        num_classes (int): Number of classes in the model (default: 5).
        save_crops_enabled (bool): If True, save cropped bounding boxes to output_base_dir.
        input_resize_factor (float): Factor to resize input images (e.g., 2.0 for 2x size, default: 2.0).
        enhance_noteheads (bool): If True, apply preprocessing to enhance noteheads (default: False).
        blur_type (str): Type of blur to apply ('gaussian', 'median', 'bilateral', or None, default: 'gaussian').
        blur_strength (int): Kernel size for blur (default: 3).
        visualize_resized (bool or str): If 'display', show resized images; if 'save', save to visualize_dir; if False, skip.
        visualize_dir (str, optional): Directory to save resized images if visualize_resized='save'.
        save_bbox_images (bool): If True, save preprocessed images with bounding boxes (no labels) to bbox_images_dir.
        bbox_images_dir (str, optional): Directory to save images with bounding boxes.

    Returns:
        dict: {group_folder: {filename: [{'bbox': [x1, y1, x2, y2], 'label': int, 'score': float}, ...]}}
              Bounding boxes sorted left-to-right (by x1) for each image, normalized to original size.
    """
    # Load model
    model = load_model(model_path, model_type=model_type, num_classes=num_classes)

    # Validate groups_path
    if not os.path.isdir(groups_path):
        print(f"⚠️ Error: {groups_path} is not a valid directory")
        return {}

    # Create output directories
    if save_crops_enabled and output_base_dir:
        os.makedirs(output_base_dir, exist_ok=True)
    if visualize_resized == 'save' and visualize_dir:
        os.makedirs(visualize_dir, exist_ok=True)
    if save_bbox_images and bbox_images_dir:
        os.makedirs(bbox_images_dir, exist_ok=True)

    # Initialize results
    results = {}

    # Get group folders
    group_folders = [f for f in os.listdir(groups_path) if
                     os.path.isdir(os.path.join(groups_path, f)) and re.match(r'group_\d+', f)]
    if not group_folders:
        print(f"⚠️ No group folders found in {groups_path}")
        return {}

    print(f"Processing {len(group_folders)} group folders: {group_folders}")

    # Process each group folder
    for group_folder in group_folders:
        input_group_dir = os.path.join(groups_path, group_folder)
        output_group_dir = os.path.join(output_base_dir, group_folder) if save_crops_enabled and output_base_dir else None
        if output_group_dir:
            os.makedirs(output_group_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_group_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"⚠️ No images found in {input_group_dir}")
            continue

        group_results = {}
        print(f"\nProcessing group: {group_folder} ({len(image_files)} images)")

        for idx, filename in enumerate(image_files):
            image_path = os.path.join(input_group_dir, filename)
            print(f"[{idx + 1}/{len(image_files)}] Processing: {filename}")

            # Load and preprocess image
            img_np, original_height, original_width = load_and_preprocess_image(
                image_path, enhance_noteheads, blur_type, blur_strength
            )
            if img_np is None:
                continue

            # Resize image
            resized_img_np, new_width, new_height = resize_image(
                img_np, input_resize_factor, original_width, original_height
            )

            # Visualize resized image
            visualize_resized_image(
                resized_img_np, filename, new_width, new_height, visualize_resized, visualize_dir
            )

            # Run inference
            predictions = run_inference(model, resized_img_np)

            # Process predictions
            filtered_predictions = process_predictions(predictions, conf_threshold, input_resize_factor)

            # Save crops if requested
            if save_crops_enabled and output_group_dir:
                img_np_original = cv2.imread(image_path)  # Load original for cropping
                save_crops(img_np_original, filename, filtered_predictions, output_group_dir)

            # Save image with bounding boxes if requested
            if save_bbox_images and bbox_images_dir:
                save_image_with_bboxes(
                    resized_img_np, filename, filtered_predictions, input_resize_factor,
                    bbox_images_dir, new_width, new_height
                )

            group_results[filename] = filtered_predictions

        results[group_folder] = group_results

    print("\n🎉 All images processed.")
    return results

def process_predict_notes(model_path, path, bbox_img_dir, output_notes_path):
    note_results = predict_notes(
        model_path=model_path,
        groups_path=path,
        output_base_dir=output_notes_path,
        conf_threshold=0.65,
        save_crops_enabled=True,
        input_resize_factor=2.0,
        enhance_noteheads=True,
        visualize_resized='display',
        blur_strength=2,
        save_bbox_images=True,
        bbox_images_dir=bbox_img_dir)

    return note_results



import os
import cv2
import numpy as np
from fastai.vision.all import *
import pathlib

# 🩹 Patch for Windows compatibility
try:
    pathlib.PosixPath = pathlib.WindowsPath
except AttributeError:
    pass

def load_fastai_model(model_path):
    model = load_learner(model_path)
    print(f"✅ Model loaded: {model_path}")
    return model

def crop_left_region(img, padding=80, min_width=150, min_size=20):
    h, w = img.shape[:2]
    x_max = min(w, min_width + padding)
    crop = img[0:h, 0:x_max]
    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        print(f"⚠️ Crop too small ({crop.shape[1]}x{crop.shape[0]}), using full image.")
        return img
    return crop

def preprocess_img_for_fastai(img, size=(640, 640)):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, size)
    return PILImage.create(img)

def setup_directories(clef_page_dir):
    base_dir = os.path.dirname(clef_page_dir)
    dirs = {
        'fClef': os.path.join(base_dir, 'fClef'),
        'gClef': os.path.join(base_dir, 'gClef'),
        'uncertain': os.path.join(base_dir, 'uncertain'),
        'cropped': os.path.join(base_dir, 'clef_cropped'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def get_clef_image_paths(clef_page_dir):
    return [
        os.path.join(clef_page_dir, f)
        for f in os.listdir(clef_page_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

def classify_clef_image(model, clef_path):
    stats = {'fClef': 0, 'gClef': 0, 'uncertain': 0, 'failed': 0}
    fname = os.path.basename(clef_path)
    print(f"📦 Classifying: {fname}")

    img = cv2.imread(clef_path)
    if img is None:
        print(f"❌ Could not read {clef_path}")
        stats['failed'] = 1
        return None, None, None, stats

    crop = crop_left_region(img)
    pil_crop = preprocess_img_for_fastai(crop)

    try:
        pred, pred_idx, probs = model.predict(pil_crop)
        label = pred if pred in ['fClef', 'gClef'] else 'uncertain'
        conf = probs[pred_idx].item()
        print(f"→ {label} ({conf:.2f})")
        stats[label] += 1
    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")
        label = 'uncertain'
        stats[label] += 1
        conf = 0.0

    return img, crop, label, stats

def save_results(clef_path, img, crop, label, stats, output_dirs, count_dict):
    base_name, ext = os.path.splitext(os.path.basename(clef_path))
    count_dict[label] += 1
    index = count_dict[label]

    # Save cropped clef image
    crop_name = f"{base_name}_crop_{label}.jpg"
    crop_path = os.path.join(output_dirs['cropped'], crop_name)
    cv2.imwrite(crop_path, crop)
    print(f"💾 Saved cropped clef to {crop_path}")

    # Save full image with updated name
    new_name = f"{base_name}_{label}{ext}"
    output_path = os.path.join(output_dirs[label], new_name)
    cv2.imwrite(output_path, img)
    print(f"💾 Saved full clef to {output_path}")

    # Remove original
    os.remove(clef_path)

def classify_and_organize_clefs(model_path, clef_root):
    model = load_fastai_model(model_path)
    global_stats = {'fClef': 0, 'gClef': 0, 'uncertain': 0, 'failed': 0}
    count_dict = {'fClef': 0, 'gClef': 0, 'uncertain': 0}

    for page in sorted(os.listdir(clef_root)):
        if not page.startswith("page_"):
            continue

        clef_page_dir = os.path.join(clef_root, page, "clef")
        if not os.path.isdir(clef_page_dir):
            print(f"🚫 No clef folder in {page}")
            continue

        output_dirs = setup_directories(clef_page_dir)
        clef_image_paths = get_clef_image_paths(clef_page_dir)

        if not clef_image_paths:
            print(f"⚠️ No clef images found in {clef_page_dir}")
            continue

        print(f"🖼️ Found {len(clef_image_paths)} clef image(s) in {page}")

        for clef_path in clef_image_paths:
            img, crop, label, stats = classify_clef_image(model, clef_path)
            for k in global_stats:
                global_stats[k] += stats[k]

            if label != 'failed':
                save_results(clef_path, img, crop, label, stats, output_dirs, count_dict)

    print("\n✅ Clefs organized successfully.")
    print("📊 Summary:", global_stats)

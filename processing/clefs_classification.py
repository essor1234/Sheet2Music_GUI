import os
import cv2
import numpy as np
from fastai.vision.all import *
import pathlib

# 🩹 Patch PosixPath for Windows compatibility with fastai
try:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
except AttributeError:
    pass

def load_fastai_model(model_path):
    model = load_learner(model_path)
    print(f"✅ Model loaded: {model_path}")
    return model

def crop_left_region(img, padding=10, min_width=100, min_size=20):
    h, w = img.shape[:2]
    x_min, y_min = 0, 0
    x_max, y_max = min(w, min_width + padding), h
    crop = img[y_min:y_max, x_min:x_max]

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

def classify_and_organize_clefs(model_path, clef_root, staff_root):
    """
    Classify clef images under clef_root/page_x/clef/*.jpg
    Move both clef and corresponding staff image into:
        - clef_root/page_x/fClef/
        - clef_root/page_x/gClef/
        - clef_root/page_x/uncertain/
    Also save cropped clef region into clef_cropped/
    """
    model = load_fastai_model(model_path)

    page_dirs = sorted([d for d in os.listdir(clef_root) if d.startswith("page_")])
    for page in page_dirs:
        clef_page_dir = os.path.join(clef_root, page, "clef")
        staff_page_dir = os.path.join(staff_root, page)

        if not os.path.exists(clef_page_dir):
            print(f"🚫 No clef folder in {page}")
            continue

        print(f"\n🔍 Processing {clef_page_dir}...")

        # Output directories
        cropped_dir = os.path.join(clef_page_dir, "..", "clef_cropped")
        os.makedirs(cropped_dir, exist_ok=True)

        class_dirs = {
            'fClef': os.path.join(clef_page_dir, "..", 'fClef'),
            'gClef': os.path.join(clef_page_dir, "..", 'gClef'),
            'uncertain': os.path.join(clef_page_dir, "..", 'uncertain'),
        }
        for d in class_dirs.values():
            os.makedirs(d, exist_ok=True)

        # Recursively search for clef images
        clef_image_paths = []
        for root, _, files in os.walk(clef_page_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    clef_image_paths.append(os.path.join(root, f))

        if not clef_image_paths:
            print(f"⚠️ No clef images found in {clef_page_dir}")
            continue

        print(f"🖼️ Found {len(clef_image_paths)} clef image(s)")

        for idx, clef_path in enumerate(clef_image_paths):
            fname = os.path.basename(clef_path)
            print(f"📦 Classifying: {fname}")

            img = cv2.imread(clef_path)
            if img is None:
                print(f"❌ Could not read {clef_path}")
                continue

            crop = crop_left_region(img)
            pil_crop = preprocess_img_for_fastai(crop)

            try:
                pred, pred_idx, probs = model.predict(pil_crop)
                label = pred if pred in ['fClef', 'gClef'] else 'uncertain'
                print(f"→ {label} ({probs[pred_idx]:.2f})")
            except Exception as e:
                print(f"⚠️ Prediction failed: {e}")
                label = 'uncertain'

            # Save cropped image
            crop_name = os.path.splitext(fname)[0] + f"_crop_{label}.jpg"
            cv2.imwrite(os.path.join(cropped_dir, crop_name), crop)

            # Move clef image to fClef/gClef/uncertain
            new_clef_path = os.path.join(class_dirs[label], fname)
            os.rename(clef_path, new_clef_path)

            # Move corresponding staffLine image
            staff_path = os.path.join(staff_page_dir, fname)
            if os.path.exists(staff_path):
                new_staff_path = os.path.join(class_dirs[label], fname)
                os.rename(staff_path, new_staff_path)

    print("\n✅ Clefs and staff lines organized successfully.")

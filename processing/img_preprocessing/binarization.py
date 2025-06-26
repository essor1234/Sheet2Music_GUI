import os
import cv2
import matplotlib.pyplot as plt

def process_binarization(path, isDisplay=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if isDisplay:
        plt.imshow(binary, cmap='gray')
        plt.title('Binarized Image')
        plt.axis('off')
        plt.show()
    return binary

def get_unique_folder(base_folder):
    """
    Returns a unique folder path by appending _1, _2, etc. if needed.
    """
    counter = 0
    candidate = base_folder
    while os.path.exists(candidate):
        counter += 1
        candidate = f"{base_folder}_{counter}"
    return candidate


def binarize_folder_images(input_folder, original_pdf_path, save_root, display=False):
    """
    Binarizes images in input_folder and saves them to a user-defined root folder.
    Folder is named after the original PDF name (e.g., sheet_binarized, sheet_binarized_1, ...).

    Args:
        input_folder (str): Folder containing images to process
        original_pdf_path (str): Path to the original PDF file (used for naming)
        save_root (str): Directory where the binarized folder should be created
        display (bool): Whether to display each binarized image

    Returns:
        str: Path to the newly created binarized folder
    """
    base_name = os.path.splitext(os.path.basename(original_pdf_path))[0]
    target_folder_name = base_name + "_binarized"
    output_folder = get_unique_folder(os.path.join(save_root, target_folder_name))

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, file)
            binary_img = process_binarization(input_path, isDisplay=display)

            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, binary_img)
            print(f"Saved binarized image to: {output_path}")

    return output_folder


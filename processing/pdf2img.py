import fitz  # PyMuPDF
import os


def get_unique_folder(base_folder):
    """
    Returns a unique folder path by appending _1, _2, ... if needed.
    """
    counter = 0
    candidate = base_folder
    while os.path.exists(candidate):
        counter += 1
        candidate = f"{base_folder}_{counter}"
    return candidate

def pdf_to_images(pdf_path, output_root, image_format='png', dpi=150):
    # Extract base name without extension
    file_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    # Create unique output folder under the root
    target_folder = get_unique_folder(os.path.join(output_root, file_stem))
    os.makedirs(target_folder, exist_ok=True)

    # Load PDF
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        image_name = f"{file_stem}_page_{i + 1}.{image_format}"
        output_path = os.path.join(target_folder, image_name)
        pix.save(output_path)
        print(f"Saved: {output_path}")

    doc.close()

    return target_folder  # Return path to access later

# Wrapper
def pdf2img(pdf_path, output_root):
    return pdf_to_images(pdf_path, output_root, dpi=400)



# def process_pdf_folder(input_folder, output_folder, image_format='png', dpi=150):
#     os.makedirs(output_folder, exist_ok=True)
#
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith('.pdf'):
#             pdf_path = os.path.join(input_folder, filename)
#             pdf_to_images(pdf_path, output_folder, image_format, dpi)



# # Example usage
# input_folder = "../DISC2"
# output_folder = "../testing/images_from_pdfs"
# process_pdf_folder(input_folder, output_folder)

# import fitz  # PyMuPDF
# import os
#
#
# def get_unique_folder(base_folder):
#     """
#     Returns a unique folder path by appending _1, _2, ... if needed.
#     """
#     counter = 0
#     candidate = base_folder
#     while os.path.exists(candidate):
#         counter += 1
#         candidate = f"{base_folder}_{counter}"
#     return candidate
#
# def pdf_to_images(pdf_path, output_root, image_format='png', dpi=150):
#     # Extract base name without extension
#     file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
#
#     # Create unique output folder under the root
#     target_folder = get_unique_folder(os.path.join(output_root, file_stem))
#     os.makedirs(target_folder, exist_ok=True)
#
#     # Load PDF
#     doc = fitz.open(pdf_path)
#
#     for i, page in enumerate(doc):
#         pix = page.get_pixmap(dpi=dpi)
#         image_name = f"{file_stem}_page_{i + 1}.{image_format}"
#         output_path = os.path.join(target_folder, image_name)
#         pix.save(output_path)
#         print(f"Saved: {output_path}")
#
#     doc.close()
#
#     return target_folder  # Return path to access later
#
# # Wrapper
# def pdf2img(pdf_path, output_root):
#     return pdf_to_images(pdf_path, output_root, dpi=400)
#
#
#

# ==============================================================

import fitz  # PyMuPDF
import os
from pathlib import Path
class PDFImageConverter:
    """
    A class to convert PDF files into images and store them in unique output directories.
    """

    def __init__(self, output_root: Path, image_format: str = "png", dpi: int = 150):
        """
        Args:
            output_root (str): Root directory where images will be saved.
            image_format (str): Format for output images (default: 'png').
            dpi (int): Resolution of rendered images (default: 150).
        """
        self.output_root = output_root
        self.image_format = image_format
        self.dpi = dpi
        os.makedirs(self.output_root, exist_ok=True)

    def _get_unique_folder(self, base_folder: str) -> str:
        """
        Generate a unique folder name by appending _1, _2, etc., if needed.

        Args:
            base_folder (str): Desired folder path.

        Returns:
            str: Unique folder path.
        """
        counter = 0
        candidate = base_folder
        while os.path.exists(candidate):
            counter += 1
            candidate = f"{base_folder}_{counter}"
        return candidate

    def convert(self, pdf_path: str) -> str:
        """
        Convert a single PDF to images and save in a unique folder.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Path to the folder containing extracted images.
        """
        file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        target_folder = self._get_unique_folder(os.path.join(self.output_root, file_stem))
        os.makedirs(target_folder, exist_ok=True)

        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=self.dpi)
            image_name = f"{file_stem}_page_{i + 1}.{self.image_format}"
            output_path = os.path.join(target_folder, image_name)
            pix.save(output_path)
            print(f"📄 Saved: {output_path}")
        doc.close()

        return target_folder

    def convert_folder(self, input_folder: str):
        """
        Convert all PDFs in a folder to images.

        Args:
            input_folder (str): Directory containing PDF files.
        """
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_folder, filename)
                print(f"\n🔄 Converting: {filename}")
                self.convert(pdf_path)

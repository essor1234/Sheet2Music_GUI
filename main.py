# from pathlib import Path
# from processing.path_setting import *
# from processing.pdf2img import *
# from processing.img_preprocessing.binarization import *
# from processing.YOLO_model_loading import *
# from processing.organizing import *
# from processing.clefs_classification import *
# from processing.staffLine_seperating import *
# from processing.grouping_process import *
# from processing.staff_localization import *
# from processing.note_localization import *
# from processing.calculate_note_ptich import *
# from processing.note_grouping import *
# from processing.musicXML_generating import *
# from processing.MXL_gen2 import *
# from processing.MXL_gen3 import *
# # Base home path (absolute)
# home_path = Path("data_storage").resolve()
#
# # Define subdirectories
#
# OG_path = home_path / "OG_path" # store original images
# binary_path = home_path / "binary_path" # store image after preprocessing
# grandStaff_path = home_path / "grandStaff_path" # store grandStaff images
# pureClefs_path = home_path / "pureClefs_path" # store clefs images
# measures_path = home_path / "measures_path" # store measure images
# cleanClefs_path = home_path / "cleanClefs_path" # store clefs images(staffLine removed)
# staffLines_path = home_path / "staffLines_path" # store staffLine images
# grouping_path = home_path/ "grouping_path" # Store clean_sep_clef and staffLine
# notations_path = home_path / "notations_path" # store notations
# results_path = home_path / "results_path" #
# note_bbox = home_path / "note_bbox"
#
# path_list = [OG_path, binary_path, staffLines_path, grandStaff_path, cleanClefs_path, measures_path, notations_path, pureClefs_path, grouping_path, note_bbox, results_path]
#
# pdf_dir = "twinkle-twinkle-little-star-piano-solo.pdf" # For testing
#
#
# grandStaff_model = "models/group_staff_seperating_2nd_v2.pt"
# clef_model = "models/clef_separating_v2.pt"
# clef_cls_model = "models/CNN_CLS/clef_doubleCheck_model_2nd.pkl"
# measure_model = "models/measure_seperating_v3.pt"
# note_predict_model = "models/fasterrcnn_finetuned_1606.pth"
#
# steps = ["C", "D", "E", "F", "G", "A", "B"]
# def pipe_line():
#     # Create requirement paths
#     create_paths(path_list)
#     # Getting image from pdf file
#     img_path = pdf2img(pdf_dir, OG_path)
#     # print(img_path)
#
#     # Pre-processing images
#     binarized_path = binarize_folder_images(img_path, pdf_dir,binary_path , display=False)
#
#     # Grand-staff separating
#     grandStaff_sep_paths = grandStaff_separating(grandStaff_model, binarized_path,pdf_dir, grandStaff_path)
#
#     # Clefs separating
#     clefs_sep_path = clefs_separating(clef_model, grandStaff_sep_paths, pureClefs_path)
#
#     # Clean clefs
#     clean_clef_crops(clefs_sep_path, verbose=True)
#
#     # StaffLine separating
#     clef_sep_path, staffLine_only_path = separate_staff_from_clefs_flat(clefs_sep_path, cleanClefs_path, staffLines_path)
#
#     # Separate into G-F clef
#     classify_and_organize_clefs(clef_cls_model, clef_sep_path)
#
#     # # Organize clefs and staffLine back to groups
#     group_all_sep_images_fixed(clef_sep_path, staffLine_only_path, grouping_path)
#     #
#     # Measure separating
#     measures_separating_from_grouping(measure_model, grouping_path, pdf_dir)
#
#     # Clean measures
#     clean_measure_crops(grouping_path,pdf_dir, verbose=True)
#
#     # Staff Localization
#     staff_results = process_group_staffs_from_grouping(grouping_path, pdf_dir)
#     # print(staff_results)
#     # Note Localization
#     note_results = process_predict_notes_from_grouping(note_predict_model, pdf_dir, grouping_path, note_bbox, notations_path)
#     # print(note_results)
#     # # Pitch Calculate
#     pitch_results = process_cal_note_pitch(note_results, staff_results, steps, print_enable=True)
#     # print(pitch_results)
#
#     final_result = group_notes_by_page_group_measure_clef(pitch_results)
#     # final_result
#     # process_musicXML_generating1(pitch_results,pdf_path=pdf_dir, output_dir=results_path, is_display=True, is_midi=True)
#     create_exact_musicxml_from_nested_results(final_result, pdf_path=pdf_dir, output_dir=results_path, is_display=True,
#                                               is_midi=True)
#     #
#     #
#
#
# # Run the pipeline
# if __name__ == "__main__":
#     pipe_line()

from pathlib import Path
from processing.path_setting import PathManager
from processing.pdf2img import PDFImageConverter
from processing.img_preprocessing.binarization import *
from processing.organizing import *
from processing.clefs_classification import *
from processing.staffLine_seperating import *
from processing.grouping_process import *
from processing.staff_localization import *
from processing.note_localization import *
from processing.calculate_note_ptich import *
from processing.note_grouping import *
from processing.musicXML_generating import *
from processing.MXL_gen2 import *
from processing.MXL_gen3 import *
from processing.yolo_pipeline import YOLODetector



# Base home path (absolute)
home_path = Path("data_storage").resolve()

# Define subdirectories
OG_path = home_path / "OG_path"  # store original images
binary_path = home_path / "binary_path"  # store image after preprocessing
grandStaff_path = home_path / "grandStaff_path"  # store grandStaff images
pureClefs_path = home_path / "pureClefs_path"  # store clefs images
measures_path = home_path / "measures_path"  # store measure images
cleanClefs_path = home_path / "cleanClefs_path"  # store clefs images(staffLine removed)
staffLines_path = home_path / "staffLines_path"  # store staffLine images
grouping_path = home_path / "grouping_path"  # Store clean_sep_clef and staffLine
notations_path = home_path / "notations_path"  # store notations
results_path = home_path / "results_path"
note_bbox = home_path / "note_bbox"

path_list = [OG_path, binary_path, staffLines_path, grandStaff_path, cleanClefs_path, measures_path,
             notations_path, pureClefs_path, grouping_path, note_bbox, results_path]

pdf_dir = "twinkle-twinkle-little-star-piano-solo.pdf"  # For testing

# Define models
grandStaff_model = "models/group_staff_seperating_2nd_v2.pt"
clef_model = "models/clef_separating_v2.pt"
clef_cls_model = "models/CNN_CLS/clef_doubleCheck_model_2nd.pkl"
measure_model = "models/measure_seperating_v3.pt"
note_predict_model = "models/fasterrcnn_finetuned_1606.pth"

steps = ["C", "D", "E", "F", "G", "A", "B"]

def pipe_line():
    # Create requirement paths
    PathManager.create(path_list)

    # Getting image from pdf file
    pdfConverter = PDFImageConverter(OG_path)
    img_path = pdfConverter.convert(pdf_dir)

    # Pre-processing images
    binarized_path = binarize_folder_images(img_path, pdf_dir, binary_path, display=False)

    # Grand-staff separating
    detector1 = YOLODetector(grandStaff_model)
    grandStaff_sep_paths = detector1.grandStaff_separating(binarized_path, pdf_dir, grandStaff_path)

    # Clefs separating
    detector2 = YOLODetector(clef_model)
    clefs_sep_path = detector2.clefs_separating(grandStaff_sep_paths, pureClefs_path)

    # Clean clefs
    clean_clef_crops(clefs_sep_path, verbose=True)

    # StaffLine separating
    clef_sep_path, staffLine_only_path = separate_staff_from_clefs_flat(clefs_sep_path, cleanClefs_path, staffLines_path)

    # Separate into G-F clef
    classify_and_organize_clefs(clef_cls_model, clef_sep_path)

    # Organize clefs and staffLine back to groups
    group_all_sep_images_fixed(clef_sep_path, staffLine_only_path, grouping_path)

    # Measure separating
    detector3 = YOLODetector(measure_model)
    detector3.measures_separating_from_grouping(grouping_path, pdf_dir)

    # Clean measures
    clean_measure_crops(grouping_path, pdf_dir, verbose=True)

    # Staff Localization
    staff_results = process_group_staffs_from_grouping(grouping_path, pdf_dir)

    # Note Localization
    note_results = process_predict_notes_from_grouping(note_predict_model, pdf_dir, grouping_path, note_bbox, notations_path)

    # Pitch Calculate
    pitch_results = process_cal_note_pitch(note_results, staff_results, steps, print_enable=True)

    # Note grouping per measure
    final_result = group_notes_by_page_group_measure_clef(pitch_results)

    # Generate MusicXML
    create_exact_musicxml_from_nested_results(final_result, pdf_path=pdf_dir, output_dir=results_path,
                                              is_display=True, is_midi=True)

# Run the pipeline
if __name__ == "__main__":
    pipe_line()

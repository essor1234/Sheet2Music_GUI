from pathlib import Path
from processing.path_setting import *
from processing.pdf2img import *
from processing.img_preprocessing.binarization import *
from processing.YOLO_model_loading import *
from processing.organizing import *
from processing.clefs_classification import *
from processing.staffLine_seperating import *
from processing.grouping_process import *
# Base home path (absolute)
home_path = Path("data_storage").resolve()

# Define subdirectories

OG_path = home_path / "OG_path" # store original images
binary_path = home_path / "binary_path" # store image after preprocessing
grandStaff_path = home_path / "grandStaff_path" # store grandStaff images
pureClefs_path = home_path / "pureClefs_path" # store clefs images
measures_path = home_path / "measures_path" # store measure images
cleanClefs_path = home_path / "cleanClefs_path" # store clefs images(staffLine removed)
staffLines_path = home_path / "staffLines_path" # store staffLine images
grouping_path = home_path/ "grouping_path" # Store clean_sep_clef and staffLine
notations_path = home_path / "notations_path" # store notations

path_list = [OG_path, binary_path, staffLines_path, grandStaff_path, cleanClefs_path, measures_path, notations_path, pureClefs_path, grouping_path]

pdf_dir = "violet_snow_for_orchestra-1.pdf" # For testing


grandStaff_model = "models/group_staff_seperating_2nd_v2.pt"
clef_model = "models/clef_separating_v2.pt"
clef_cls_model = "models/CNN_CLS/clef_doubleCheck_model_2nd.pkl"
measure_model = "models/measure_seperating_v3.pt"
def pipe_line():
    # Create requirement paths
    create_paths(path_list)
    # Getting image from pdf file
    img_path = pdf2img(pdf_dir, OG_path)
    print(img_path)

    # Pre-processing images
    binarized_path = binarize_folder_images(img_path, pdf_dir,binary_path , display=False)

    # Grand-staff separating
    grandStaff_sep_paths = grandStaff_separating(grandStaff_model, binarized_path,pdf_dir, grandStaff_path)

    # Clefs separating
    clefs_sep_path = clefs_separating(clef_model, grandStaff_sep_paths, pureClefs_path)

    # Clean clefs
    clean_clef_crops(clefs_sep_path, verbose=True)

    # StaffLine seperating
    clef_sep_path, staffLine_only_path = separate_staff_from_clefs_flat(clefs_sep_path, cleanClefs_path, staffLines_path)

    # Separate into G-F clef
    classify_and_organize_clefs(clef_cls_model, clef_sep_path, staffLine_only_path)

    # Organize clefs and staffLine back to groups
    group_all_sep_images_fixed(clef_sep_path, staffLine_only_path, grouping_path)

    # Measure separating
    measures_separating_from_grouping(measure_model, grouping_path)

    # Clean measures
    clean_measure_crops(grouping_path, verbose=True)

# Run the pipeline
if __name__ == "__main__":
    pipe_line()
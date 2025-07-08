# import math
# import re
#
#
#
# n_ref = 1 # E4 or G2
# steps = ["C", "D", "E", "F", "G", "A", "B"]
#
#
# import os
#
# def parse_folder_info_from_filename(filename, grouping_path):
#     """
#     Extract the PDF name, page, group, and clefs directory from filename.
#
#     Args:
#         filename (str): Full filename
#         grouping_path (str): Root path to grouping folder
#
#     Returns:
#         full_group_path (str): Path like grouping_path/pdf_name/page_x/group_x/clefs
#     """
#     # Match example: violet_snow_for_orchestra-1_page_1_staff_group_0_clef_1_gClef_1_measure_0.jpg
#     match = re.match(r'(.+?)_page_(\d+)_staff_group_(\d+)_.*', filename)
#     if not match:
#         raise ValueError(f"Filename structure is invalid: {filename}")
#     pdf_name = match.group(1)
#     page_num = match.group(2)
#     group_num = match.group(3)
#
#     full_group_path = os.path.join(
#         grouping_path,
#         pdf_name,
#         f"page_{page_num}",
#         f"group_{group_num}",
#         "clefs"
#     )
#     return full_group_path
#
# def cal_bouding_box_center(bbox):
#     """
#     Calculate the center of a bounding box with floating-point precision.
#
#     Args:
#         bbox (list): [xMin, yMin, xMax, yMax]
#
#     Returns:
#         tuple: (x_center, y_center) as floats
#     """
#     xMin, yMin, xMax, yMax = bbox[0], bbox[1], bbox[2], bbox[3]
#     x_center = (xMin + xMax) / 2
#     y_center = (yMin + yMax) / 2
#     return (x_center, y_center)
#
#
# # def cal_n_position(lastCoor, y_center, distance):
# #     """
# #     Calculate note position relative to the bottom staff line with floating-point precision.
# #
# #     Args:
# #         lastCoor (float): Y-coordinate of the bottom staff line
# #         y_center (float): Y-center of the note's bounding box
# #         distance (float): Average distance between staff lines
# #
# #     Returns:
# #         float: Position relative to the staff
# #     """
# #     position = (y_center - (lastCoor + distance / 2)) / (distance / 2) * -1
# #     return position
#
# def cal_n_position(lastCoor, y_center, distance):
#     position = (lastCoor + distance / 2 - y_center) / (distance / 2)
#     return position
#
#
# def cal_step_num(n_note, n_ref):
#     """
#     Calculate the step number difference.
#
#     Args:
#         n_note (float): Snapped note position
#         n_ref (float): Reference position
#
#     Returns:
#         int: Step number difference
#     """
#     return int(n_note - n_ref)
#
#
# def get_step_idx(steps, step_num, ref_idx):
#     """
#     Get the step index (note letter) based on step number and reference index.
#
#     Args:
#         steps (list): List of note letters, e.g., ["C", "D", "E", "F", "G", "A", "B"]
#         step_num (int): Step number difference
#         ref_idx (int): Reference index in steps
#
#     Returns:
#         str: Note letter
#     """
#     step_index = int((ref_idx + step_num) % 7)
#     return steps[step_index]
#
#
# def cal_octave(step_num, ref_idx, clef_type):
#     """
#     Calculate octave number from step difference and reference index.
#     """
#     base_octave = 4 if clef_type == 'gClef' else 2
#     return base_octave + math.floor((ref_idx + step_num) // 7)
#
#
# def get_clef_type_and_base_filename(filename):
#     """
#     Extract clef type and base staff filename from a note filename.
#     Works with filenames like:
#     - '..._clef_0_gClef_measure_0.jpg'
#     - '..._clef_0_gClef_1_measure_0.jpg'
#
#     Returns:
#         - clef type (e.g., 'gClef', 'fClef', or 'unknown')
#         - base filename of the staff image to match (e.g., '..._clef_0_gClef.jpg')
#     """
#     match = re.match(r'(.+_clef_\d+)_([a-zA-Z]+)(?:_\d+)?_measure_\d+\.jpg$', filename)
#     if match:
#         base_filename = f"{match.group(1)}_{match.group(2)}.jpg"
#         clef_type_raw = match.group(2).lower()
#         if clef_type_raw == 'gclef':
#             return 'gClef', base_filename
#         elif clef_type_raw == 'fclef':
#             return 'fClef', base_filename
#         else:
#             print(f"⚠️ Unknown clef type '{clef_type_raw}' in {filename}")
#             return 'unknown', base_filename
#     else:
#         print(f"⚠️ Filename doesn't match expected clef structure: {filename}")
#         return 'unknown', filename
#
#
#
#
#
#
# def calculate_note_pitches(note_results, staff_results, steps=None, default_n_ref=1, default_ref_idx=2):
#     if steps is None:
#         steps = ["C", "D", "E", "F", "G", "A", "B"]
#     pitch_results = {}
#
#     for group in note_results:
#         if group not in staff_results:
#             print(f"⚠️ Group {group} not found in staff results, skipping")
#             continue
#
#         pitch_results[group] = {}
#
#         for filename, note_preds in note_results[group].items():
#             if filename.endswith('_staff_lines.jpg'):
#                 continue
#
#             clef_type, base_filename = get_clef_type_and_base_filename(filename)
#
#             if clef_type == 'gClef':
#                 n_ref = 1  # E4 on bottom line
#                 ref_idx = 2  # E
#             elif clef_type == 'fClef':
#                 n_ref = 1  # G2 on bottom line
#                 ref_idx = 4  # G
#             else:
#                 n_ref = default_n_ref
#                 ref_idx = default_ref_idx
#
#             if base_filename not in staff_results[group]:
#                 print(f"⚠️ Base image {base_filename} for {filename} not found in staff results[{group}], skipping")
#                 continue
#
#             staff_lines = staff_results[group][base_filename]['staff_lines']
#             if not staff_lines or len(staff_lines) < 2:
#                 print(f"⚠️ Insufficient staff lines for {base_filename}, skipping")
#                 continue
#
#             staff_lines_sorted = sorted(staff_lines, key=lambda x: x[1], reverse=True)
#             lastCoor = staff_lines_sorted[0][1]
#             firstCoor = staff_lines_sorted[-1][1]
#             distance = int((lastCoor - firstCoor) / 4)
#
#             note_pitches = []
#             for note in note_preds:
#                 bbox = note['bbox']
#                 label = note['label']
#                 score = note['score']
#
#                 _, y_center = cal_bouding_box_center(bbox)
#                 position = cal_n_position(lastCoor, y_center, distance)
#
#                 n_note = round(position)
#                 step_num = cal_step_num(n_note, n_ref)
#                 step = get_step_idx(steps, step_num, ref_idx)
#                 octave = cal_octave(step_num, ref_idx, clef_type)
#
#                 note_pitches.append({
#                     'bbox': bbox,
#                     'label': label,
#                     'score': score,
#                     'step': step,
#                     'octave': octave
#                 })
#
#             pitch_results[group][filename] = note_pitches
#
#     return pitch_results
#
#
#
# def process_cal_note_pitch(note_results, staff_results, steps, print_enable=False):
#     pitch_results = calculate_note_pitches(
#         note_results=note_results,
#         staff_results=staff_results,
#         steps=steps,
#         default_n_ref=1,
#         default_ref_idx=2
#     )
#
#     if print_enable:
#         for group, images in pitch_results.items():
#             print(f"\nGroup: {group}")
#             for filename, notes in images.items():
#                 print(f"  Image: {filename}")
#                 for i, note in enumerate(notes):
#                     print(f"    Note {i}: bbox={note['bbox']}, label={note['label']}, "
#                           f"score={note['score']:.2f}, pitch={note['step']}{note['octave']}")
#
#     return pitch_results
#
# =====================================================================
import math
import re
import os


class NotePitchCalculator:
    def __init__(self, steps=None, default_n_ref=1, default_ref_idx=2):
        self.steps = steps if steps else ["C", "D", "E", "F", "G", "A", "B"]
        self.default_n_ref = default_n_ref
        self.default_ref_idx = default_ref_idx

    @staticmethod
    def parse_folder_info_from_filename(filename, grouping_path):
        match = re.match(r'(.+?)_page_(\d+)_staff_group_(\d+)_.*', filename)
        if not match:
            raise ValueError(f"Filename structure is invalid: {filename}")
        pdf_name = match.group(1)
        page_num = match.group(2)
        group_num = match.group(3)

        full_group_path = os.path.join(
            grouping_path,
            pdf_name,
            f"page_{page_num}",
            f"group_{group_num}",
            "clefs"
        )
        return full_group_path

    @staticmethod
    def cal_bounding_box_center(bbox):
        xMin, yMin, xMax, yMax = bbox
        return ((xMin + xMax) / 2, (yMin + yMax) / 2)

    @staticmethod
    def cal_n_position(lastCoor, y_center, distance):
        return (lastCoor + distance / 2 - y_center) / (distance / 2)

    @staticmethod
    def cal_step_num(n_note, n_ref):
        return int(n_note - n_ref)

    def get_step_idx(self, step_num, ref_idx):
        return self.steps[(ref_idx + step_num) % 7]

    @staticmethod
    def cal_octave(step_num, ref_idx, clef_type):
        base_octave = 4 if clef_type == 'gClef' else 2
        return base_octave + math.floor((ref_idx + step_num) / 7)

    @staticmethod
    def get_clef_type_and_base_filename(filename):
        match = re.match(r'(.+_clef_\d+)_([a-zA-Z]+)(?:_\d+)?_measure_\d+\.jpg$', filename)
        if match:
            base_filename = f"{match.group(1)}_{match.group(2)}.jpg"
            clef_type_raw = match.group(2).lower()
            if clef_type_raw == 'gclef':
                return 'gClef', base_filename
            elif clef_type_raw == 'fclef':
                return 'fClef', base_filename
            else:
                print(f"⚠️ Unknown clef type '{clef_type_raw}' in {filename}")
                return 'unknown', base_filename
        else:
            print(f"⚠️ Filename doesn't match expected clef structure: {filename}")
            return 'unknown', filename

    def calculate_note_pitches(self, note_results, staff_results):
        pitch_results = {}

        for group in note_results:
            if group not in staff_results:
                print(f"⚠️ Group {group} not found in staff results, skipping")
                continue

            pitch_results[group] = {}

            for filename, note_preds in note_results[group].items():
                if filename.endswith('_staff_lines.jpg'):
                    continue

                clef_type, base_filename = self.get_clef_type_and_base_filename(filename)

                if clef_type == 'gClef':
                    n_ref = 1
                    ref_idx = 2  # E
                elif clef_type == 'fClef':
                    n_ref = 1
                    ref_idx = 4  # G
                else:
                    n_ref = self.default_n_ref
                    ref_idx = self.default_ref_idx

                if base_filename not in staff_results[group]:
                    print(f"⚠️ Base image {base_filename} for {filename} not found in staff results[{group}], skipping")
                    continue

                staff_lines = staff_results[group][base_filename]['staff_lines']
                if not staff_lines or len(staff_lines) < 2:
                    print(f"⚠️ Insufficient staff lines for {base_filename}, skipping")
                    continue

                staff_lines_sorted = sorted(staff_lines, key=lambda x: x[1], reverse=True)
                lastCoor = staff_lines_sorted[0][1]
                firstCoor = staff_lines_sorted[-1][1]
                distance = (lastCoor - firstCoor) / 4

                note_pitches = []
                for note in note_preds:
                    bbox = note['bbox']
                    label = note['label']
                    score = note['score']

                    _, y_center = self.cal_bounding_box_center(bbox)
                    position = self.cal_n_position(lastCoor, y_center, distance)

                    n_note = round(position)
                    step_num = self.cal_step_num(n_note, n_ref)
                    step = self.get_step_idx(step_num, ref_idx)
                    octave = self.cal_octave(step_num, ref_idx, clef_type)

                    note_pitches.append({
                        'bbox': bbox,
                        'label': label,
                        'score': score,
                        'step': step,
                        'octave': octave
                    })

                pitch_results[group][filename] = note_pitches

        return pitch_results

    def process(self, note_results, staff_results, print_enable=False):
        pitch_results = self.calculate_note_pitches(note_results, staff_results)

        if print_enable:
            for group, images in pitch_results.items():
                print(f"\nGroup: {group}")
                for filename, notes in images.items():
                    print(f"  Image: {filename}")
                    for i, note in enumerate(notes):
                        print(f"    Note {i}: bbox={note['bbox']}, label={note['label']}, "
                              f"score={note['score']:.2f}, pitch={note['step']}{note['octave']}")

        return pitch_results

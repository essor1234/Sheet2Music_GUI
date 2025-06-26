import math
import re



n_ref = 1 # E4 or G2
steps = ["C", "D", "E", "F", "G", "A", "B"]



def cal_bouding_box_center(bbox):
    """
    Calculate the center of a bounding box with floating-point precision.

    Args:
        bbox (list): [xMin, yMin, xMax, yMax]

    Returns:
        tuple: (x_center, y_center) as floats
    """
    xMin, yMin, xMax, yMax = bbox[0], bbox[1], bbox[2], bbox[3]
    x_center = (xMin + xMax) / 2
    y_center = (yMin + yMax) / 2
    return (x_center, y_center)


def cal_n_position(lastCoor, y_center, distance):
    """
    Calculate note position relative to the bottom staff line with floating-point precision.

    Args:
        lastCoor (float): Y-coordinate of the bottom staff line
        y_center (float): Y-center of the note's bounding box
        distance (float): Average distance between staff lines

    Returns:
        float: Position relative to the staff
    """
    position = (lastCoor + distance / 2 - y_center) / (distance / 2)
    return position


def cal_step_num(n_note, n_ref):
    """
    Calculate the step number difference.

    Args:
        n_note (float): Snapped note position
        n_ref (float): Reference position

    Returns:
        int: Step number difference
    """
    return int(n_note - n_ref)


def get_step_idx(steps, step_num, ref_idx):
    """
    Get the step index (note letter) based on step number and reference index.

    Args:
        steps (list): List of note letters, e.g., ["C", "D", "E", "F", "G", "A", "B"]
        step_num (int): Step number difference
        ref_idx (int): Reference index in steps

    Returns:
        str: Note letter
    """
    step_index = int((ref_idx + step_num) % 7)
    return steps[step_index]


def cal_octave(step_num, ref_idx, clef_type):
    """
    Calculate octave number from step difference and reference index.
    """
    base_octave = 4 if clef_type == 'gClef' else 2
    return base_octave + math.floor((ref_idx + step_num) // 7)


def get_clef_type_and_base_filename(filename):
    measure_match = re.match(r'(.+?)(?:_measure_\d+)?\.jpg$', filename)
    base_filename = measure_match.group(1) + '.jpg' if measure_match else filename
    filename_lower = filename.lower()
    if 'gclef' in filename_lower:
        return 'gClef', base_filename
    elif 'fclef' in filename_lower:
        return 'fClef', base_filename
    print(f"Warning: Unknown clef type for {filename}")
    return 'unknown', base_filename


def calculate_note_pitches(note_results, staff_results, steps=None, default_n_ref=1,
                           default_ref_idx=2):
    """
    Calculate pitch for each note using bounding box and staff line coordinates.

    Returns:
        dict: {group_folder: {filename: [{'bbox': ..., 'step': ..., 'octave': ...}, ...]}}
    """
    if steps is None:
        steps = ["C", "D", "E", "F", "G", "A", "B"]
    pitch_results = {}

    for group in note_results:
        if group not in staff_results:
            print(f"⚠️ Group {group} not found in staff results, skipping")
            continue

        pitch_results[group] = {}

        for filename, note_preds in note_results[group].items():
            if filename.endswith('_staff_lines.jpg'):
                continue

            clef_type, base_filename = get_clef_type_and_base_filename(filename)

            if clef_type == 'gClef':
                n_ref = 1  # Reference position for E4
                ref_idx = 2  # E in steps
            elif clef_type == 'fClef':
                n_ref = 1  # Reference position for G2
                ref_idx = 4  # G in steps
            else:
                n_ref = default_n_ref
                ref_idx = default_ref_idx

            if base_filename not in staff_results[group]:
                print(f"⚠️ Base image {base_filename} for {filename} not found in staff results, skipping")
                continue

            staff_lines = staff_results[group][base_filename]['staff_lines']
            if not staff_lines or len(staff_lines) < 2:
                print(f"⚠️ Insufficient staff lines for {base_filename}, skipping")
                continue

            # Ensure lastCoor is the bottom line (largest y), firstCoor is the top line (smallest y)
            staff_lines_sorted = sorted(staff_lines, key=lambda x: x[1], reverse=True)
            lastCoor = staff_lines_sorted[0][1]  # Bottom line (largest y)
            firstCoor = staff_lines_sorted[-1][1]  # Top line (smallest y)
            distance = int((lastCoor - firstCoor) / 4)

            note_pitches = []
            for note in note_preds:
                bbox = note['bbox']
                label = note['label']
                score = note['score']

                _, y_center = cal_bouding_box_center(bbox)
                position = cal_n_position(lastCoor, y_center, distance)

                n_note = round(position)  # Snap to nearest integer n
                step_num = cal_step_num(n_note, n_ref)
                step = get_step_idx(steps, step_num, ref_idx)
                octave = cal_octave(step_num, ref_idx, clef_type)

                note_pitches.append({
                    'bbox': bbox,
                    'label': label,
                    'score': score,
                    'step': step,
                    'octave': octave
                })

            pitch_results[group][filename] = note_pitches

    return pitch_results


def process_cal_note_pitch(note_results, staff_results, steps, print_enable=False):
    pitch_results = calculate_note_pitches(
        note_results=note_results,
        staff_results=staff_results,
        steps=steps,
        default_n_ref=1,  # Default for G-clef
        default_ref_idx=2,  # Default for G-clef
    )

    if print_enable:
        # Step 4: Print results
        for group, images in pitch_results.items():
            print(f"\nGroup: {group}")
            for filename, notes in images.items():
                print(f"  Image: {filename}")
                for i, note in enumerate(notes):
                    print(f"    Note {i}: bbox={note['bbox']}, label={note['label']}, "
                          f"score={note['score']:.2f}, pitch={note['step']}{note['octave']}")

    return pitch_results
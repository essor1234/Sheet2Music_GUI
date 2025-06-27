import re
from collections import defaultdict


def group_notes_across_groups_by_measure_and_x(pitch_results, x_diff_threshold=20):
    """
    Groups notes across all groups and clefs by group, measure, and x-coordinate for chords.

    Args:
        pitch_results (dict): Dictionary of group data, e.g., pitch_results['group_0'] = {filename: notes}
        x_diff_threshold (float): Maximum x difference to cluster notes as a chord

    Returns:
        dict: {
            group_id: {
                'measures': {measure_num: [{'x_start': float, 'notes': [{'staff': int, 'step': str, 'octave': int, 'duration': int, 'type': str}]}]},
                'clefs': [(clef_idx, clef_type), ...]  # e.g., [('0', 'fClef'), ('1', 'fClef')]
            }
        }
    """
    grouped_notes = defaultdict(lambda: defaultdict(list))
    clefs_by_group = defaultdict(list)
    filename_pattern = r'_group_(\d+)_clef_(\d+)_(gClef|fClef)_(\d+)_measure_(\d+)\.jpg'

    for group_id, images in pitch_results.items():
        for filename, notes in images.items():
            match = re.search(filename_pattern, filename)
            if not match:
                print(f"Warning: Filename {filename} does not match pattern, skipping.")
                continue
            group_idx, clef_idx, clef_type, _, measure_num = match.groups()
            measure_num = int(measure_num)
            staff = 1 if clef_idx == '0' else 2

            # Store clef info
            if (clef_idx, clef_type) not in clefs_by_group[group_id]:
                clefs_by_group[group_id].append((clef_idx, clef_type))

            for note in notes:
                x_start = note['bbox'][0]
                # Default to quarter note if duration/type not provided
                duration = note.get('duration', 1)
                note_type = note.get('type', 'quarter')
                grouped_notes[group_id][measure_num].append({
                    'x_start': x_start,
                    'notes': [{
                        'staff': staff,
                        'step': note['step'],
                        'octave': note['octave'],
                        'duration': duration,
                        'type': note_type
                    }]
                })

        # Ensure two clefs per group (default to F-clef for missing clef)
        if len(clefs_by_group[group_id]) < 2:
            missing_clef_idx = '1' if '0' in [c[0] for c in clefs_by_group[group_id]] else '0'
            clefs_by_group[group_id].append((missing_clef_idx, 'fClef'))

    # Cluster notes by x_start
    clustered_groups = {}
    for group_id, measures in grouped_notes.items():
        clustered_measures = {}
        for measure_num, note_entries in measures.items():
            if not note_entries:
                continue

            # Sort by x_start
            note_entries.sort(key=lambda x: x['x_start'])
            clustered = []
            current_cluster = [note_entries[0]]

            for note in note_entries[1:]:
                if note['x_start'] - current_cluster[-1]['x_start'] <= x_diff_threshold:
                    current_cluster.append(note)
                else:
                    # Combine notes into a single time slot
                    clustered.append({
                        'x_start': current_cluster[0]['x_start'],
                        'notes': [n['notes'][0] for n in current_cluster]
                    })
                    current_cluster = [note]

            if current_cluster:
                clustered.append({
                    'x_start': current_cluster[0]['x_start'],
                    'notes': [n['notes'][0] for n in current_cluster]
                })

            clustered_measures[measure_num] = clustered
        clustered_groups[group_id] = {
            'measures': clustered_measures,
            'clefs': sorted(clefs_by_group[group_id], key=lambda x: x[0])
        }

    return clustered_groups
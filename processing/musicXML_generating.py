import re
from typing import Dict, List, Any
from music21 import converter, environment


def parse_measure_number(filename: str) -> int:
    """Extract measure number from filename."""
    try:
        match = re.search(r'measure_(\d+)', filename)
        return int(match.group(1)) if match else 0
    except Exception as e:
        print(f"Warning: Failed to parse measure number from {filename}: {e}")
        return 0


def parse_clef_info(filename: str) -> tuple[str, int]:
    """Extract clef type and staff number from filename."""
    try:
        if 'gClef' in filename:
            match = re.search(r'gClef_(\d+)', filename)
            clef_index = int(match.group(1)) if match else 0
            return 'G', 1 if clef_index <= 1 else 2  # gClef_1 or lower → staff 1, else staff 2
        elif 'fClef' in filename:
            match = re.search(r'fClef_(\d+)', filename)
            clef_index = int(match.group(1)) if match else 0
            return 'F', 1 if clef_index <= 1 else 2  # fClef_1 → staff 1, fClef_2 → staff 2
        else:
            print(f"Warning: Unknown clef in {filename}, defaulting to G-clef")
            return 'G', 1
    except Exception as e:
        print(f"Warning: Error parsing clef from {filename}: {e}")
        return 'G', 1


def validate_note_data(note_data: Dict[str, Any]) -> bool:
    """Validate that note data contains required fields."""
    required = ['step', 'octave', 'bbox']
    return all(key in note_data for key in required) and isinstance(note_data['bbox'], list) and len(
        note_data['bbox']) >= 1


def organize_notes(pitch_results: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, List[List[Dict[str, Any]]]]]]:
    """Organize notes by group, measure, and staff, grouping close notes as chords."""
    organized_data = {}
    chord_threshold = 30.0

    for group_id, group_data in pitch_results.items():
        if not isinstance(group_data, dict):
            print(f"Warning: Skipping invalid group {group_id}: expected dict, got {type(group_data)}")
            continue

        organized_data[group_id] = {}

        for filename, notes in group_data.items():
            if not isinstance(notes, list):
                print(f"Warning: Skipping invalid notes for {filename}: expected list, got {type(notes)}")
                continue

            measure_num = parse_measure_number(filename)
            clef_type, staff_num = parse_clef_info(filename)

            if measure_num not in organized_data[group_id]:
                organized_data[group_id][measure_num] = {'treble': [], 'bass': []}

            note_infos = []
            for note_data in notes:
                if not validate_note_data(note_data):
                    print(f"Warning: Skipping invalid note in {filename}: {note_data}")
                    continue
                note_info = {
                    'step': note_data['step'],
                    'octave': note_data['octave'],
                    'bbox_x': float(note_data['bbox'][0]),
                    'score': note_data.get('score', 0.0)
                }
                note_infos.append(note_info)

            note_infos.sort(key=lambda x: x['bbox_x'])

            chords = []
            current_chord = []
            for note_info in note_infos:
                if not current_chord:
                    current_chord.append(note_info)
                else:
                    last_x = current_chord[-1]['bbox_x']
                    current_x = note_info['bbox_x']
                    if current_x - last_x <= chord_threshold:
                        current_chord.append(note_info)
                    else:
                        chords.append(current_chord)
                        current_chord = [note_info]
            if current_chord:
                chords.append(current_chord)

            if staff_num == 1:
                organized_data[group_id][measure_num]['treble'].extend(chords)
            else:
                organized_data[group_id][measure_num]['bass'].extend(chords)

    return organized_data


def create_exact_musicxml(pitch_results: Dict[str, Any]) -> str:
    """Create MusicXML with grand staves for each group, handling chords and dynamic clefs."""
    organized_data = organize_notes(pitch_results)

    musicxml_content = '''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work>
    <work-title>Grand Staves from Pitch Results</work-title>
  </work>
  <identification>
    <creator type="composer">Custom</creator>
    <encoding>
      <encoding-date>2025-06-22</encoding-date>
      <software>Python</software>
    </encoding>
  </identification>
  <defaults>
    <scaling>
      <millimeters>7</millimeters>
      <tenths>40</tenths>
    </scaling>
  </defaults>
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">
'''

    if not organized_data:
        print("Warning: No valid groups found in pitch_results, adding empty measure")
        musicxml_content += '''    <measure number="1">
      <attributes>
        <divisions>10080</divisions>
        <key>
          <fifths>0</fifths>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <staves>2</staves>
        <clef number="1">
          <sign>G</sign>
          <line>2</line>
        </clef>
        <clef number="2">
          <sign>F</sign>
          <line>4</line>
        </clef>
      </attributes>
      <note>
        <rest/>
        <duration>40320</duration>
        <voice>1</voice>
        <type>whole</type>
        <staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>'''
        return musicxml_content

    measure_number = 1
    group_ids = sorted(organized_data.keys(),
                       key=lambda x: int(x.split('_')[-1]) if x.startswith('group_') else float('inf'))

    for group_index, group_id in enumerate(group_ids):
        group_measures = sorted(organized_data[group_id].keys())
        max_measure = max(group_measures) if group_measures else -1

        clef_staff_1 = 'G'
        clef_staff_2 = 'F'
        for filename in pitch_results.get(group_id, {}):
            clef_type, staff_num = parse_clef_info(filename)
            if staff_num == 1:
                clef_staff_1 = clef_type
            elif staff_num == 2:
                clef_staff_2 = clef_type

        musicxml_content += f'''    <!--========================= {group_id}: Grand Staff {group_index + 1} ==========================-->
    <measure number="{measure_number}">
'''
        if group_index > 0:
            musicxml_content += '''      <print new-system="yes"/>
'''
        musicxml_content += f'''      <attributes>
        <divisions>10080</divisions>
        <key>
          <fifths>0</fifths>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <staves>2</staves>
        <clef number="1">
          <sign>{clef_staff_1}</sign>
          <line>{2 if clef_staff_1 == 'G' else 4}</line>
        </clef>
        <clef number="2">
          <sign>{clef_staff_2}</sign>
          <line>{4 if clef_staff_2 == 'F' else 2}</line>
        </clef>
      </attributes>
      <direction placement="above">
        <direction-type>
          <words font-weight="bold">{group_id.replace('_', ' ').title()}</words>
        </direction-type>
      </direction>
'''

        for measure_num in range(max_measure + 1):
            notes = organized_data[group_id].get(measure_num, {'treble': [], 'bass': []})
            treble_chords = sorted(notes['treble'], key=lambda x: x[0]['bbox_x'] if x else float('inf'))
            bass_chords = sorted(notes['bass'], key=lambda x: x[0]['bbox_x'] if x else float('inf'))

            if measure_num > 0:
                musicxml_content += f'''    <measure number="{measure_number}">
'''

            for chord in treble_chords:
                for i, note_data in enumerate(chord):
                    try:
                        musicxml_content += '''      <note>
'''
                        if i > 0:
                            musicxml_content += '''        <chord/>
'''
                        musicxml_content += f'''        <pitch>
          <step>{note_data['step']}</step>
          <octave>{int(note_data['octave'])}</octave>
        </pitch>
        <duration>20160</duration>
        <voice>1</voice>
        <type>half</type>
        <staff>1</staff>
      </note>
'''
                    except (ValueError, KeyError) as e:
                        print(
                            f"Warning: Skipping invalid note in {group_id}, measure {measure_num}, treble: {note_data}, error: {e}")

            for chord in bass_chords:
                for i, note_data in enumerate(chord):
                    try:
                        musicxml_content += '''      <note>
'''
                        if i > 0:
                            musicxml_content += '''        <chord/>
'''
                        musicxml_content += f'''        <pitch>
          <step>{note_data['step']}</step>
          <octave>{int(note_data['octave'])}</octave>
        </pitch>
        <duration>20160</duration>
        <voice>2</voice>
        <type>half</type>
        <staff>2</staff>
      </note>
'''
                    except (ValueError, KeyError) as e:
                        print(
                            f"Warning: Skipping invalid note in {group_id}, measure {measure_num}, bass: {note_data}, error: {e}")

            if not treble_chords and not bass_chords:
                musicxml_content += '''      <note>
        <rest/>
        <duration>40320</duration>
        <voice>1</voice>
        <type>whole</type>
        <staff>1</staff>
      </note>
'''

            musicxml_content += '''    </measure>
'''
            measure_number += 1

        if group_index < len(group_ids) - 1:
            musicxml_content = musicxml_content.rstrip('</measure>\n') + '''      <barline location="right">
        <bar-style>light-heavy</bar-style>
      </barline>
    </measure>
'''

    musicxml_content += '''  </part>
</score-partwise>'''

    return musicxml_content


def save_musicxml_file(content: str, filename: str = 'pitch_results_musicxml.musicxml') -> bool:
    """Save the MusicXML content to a .musicxml file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ MusicXML file '{filename}' has been created successfully!")
        print(f"  File size: {len(content)} characters")
        print(f"  Extension: {filename.split('.')[-1]}")

        lines = content.split('\n')
        part_p1 = '<part id="P1">'
        part_p2 = '<part id="P2">'

        return True

    except Exception as e:
        print(f"✗ Error saving MusicXML file: {e}")
        return False


def display_or_play_score(musicxml_content: str, mode: str = 'auto', filename: str = 'temp_musicxml.musicxml') -> bool:
    """Display or play the MusicXML score based on specified mode."""
    try:
        env = environment.Environment()
        env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

        if not save_musicxml_file(musicxml_content, filename):
            print("Error: Failed to save temporary MusicXML file")
            return False

        score = converter.parse(filename)

        notation_software = env['musicxmlPath']
        has_notation_software = notation_software is not None and notation_software.exists()

        if mode == 'auto':
            if has_notation_software:
                print("Auto mode: Notation software detected, displaying score visually")
                score.show()
            else:
                print("Auto mode: No notation software detected, playing MIDI")
                score.show('midi')
        elif mode == 'notation':
            if has_notation_software:
                print("Displaying score visually")
                score.show()
            else:
                print("Error: No notation software configured. Ensure MuseScore path is correct")
                return False
        elif mode == 'midi':
            print("Playing score as MIDI")
            score.show('midi')
        else:
            print(f"Error: Invalid mode '{mode}'. Use 'notation', 'midi', or 'auto'")
            return False

        return True

    except Exception as e:
        print(f"Error in display_or_play_score: {e}")
        return False



def process_musicXML_generating(pitch_results,file_name="temp_musicxml.musicxml", is_display=True, is_midi=True):
    exact_xml = create_exact_musicxml(pitch_results)
    if save_musicxml_file(exact_xml, file_name):
        print("\n🎵 MusicXML file with pitch results created successfully!")
        if is_display:
            display_or_play_score(exact_xml, mode='notation')
        if is_midi:
            display_or_play_score(exact_xml, mode='midi')


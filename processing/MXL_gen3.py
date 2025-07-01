import re
from typing import Dict, Any
from music21 import converter, environment
import os
from pathlib import Path
from datetime import datetime

def create_exact_musicxml_from_nested_results(
    nested_results: Dict[str, Dict[str, Dict[str, Dict[str, list]]]],
    pdf_path: str = None,
    output_dir: str = "results",
    is_display: bool = True,
    is_midi: bool = True,
    chord_threshold: float = 30.0
):
    def get_staff_num_and_clef(clef_name: str) -> tuple:
        if clef_name.lower() == 'gclef':
            return 1, 'G'
        elif clef_name.lower() == 'fclef':
            return 2, 'F'
        else:
            return 1, 'G'  # Default

    def get_line_for_clef(clef_sign: str) -> int:
        return 2 if clef_sign == 'G' else 4

    # === Setup Output Path ===
    if pdf_path:
        pdf_name = Path(pdf_path).stem
    else:
        first_key = next(iter(nested_results))
        pdf_match = re.match(r'([^_]+(?:_[^_]+)*-\d+)_page_\d+_', first_key)
        pdf_name = pdf_match.group(1) if pdf_match else "unknown_pdf"

    save_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(save_dir, exist_ok=True)

    musicxml_filename = os.path.join(save_dir, f"{pdf_name}_results.musicxml")
    midi_filename = os.path.join(save_dir, f"{pdf_name}_results.mid")

    now = datetime.now().strftime("%Y-%m-%d")

    # === Begin MusicXML content ===
    musicxml_content = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work>
    <work-title>Generated Score</work-title>
  </work>
  <identification>
    <creator type="composer">Sheet2Music</creator>
    <encoding>
      <encoding-date>{now}</encoding-date>
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

    measure_counter = 1
    for page_id in sorted(nested_results.keys()):
        page_data = nested_results[page_id]
        for group_id in sorted(page_data.keys()):
            group_data = page_data[group_id]
            for measure_id in sorted(group_data.keys(), key=lambda x: int(re.search(r'\d+', x).group())):
                clef_dict = group_data[measure_id]

                musicxml_content += f'''    <measure number="{measure_counter}">
'''
                if measure_counter == 1:
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
'''
                    clef_done = set()
                    for clef_name in clef_dict:
                        staff_num, clef_sign = get_staff_num_and_clef(clef_name)
                        if staff_num not in clef_done:
                            musicxml_content += f'''        <clef number="{staff_num}">
          <sign>{clef_sign}</sign>
          <line>{get_line_for_clef(clef_sign)}</line>
        </clef>
'''
                            clef_done.add(staff_num)
                    musicxml_content += f'''      </attributes>
'''

                for clef_name, clef_notes in clef_dict.items():
                    staff_num, _clef_sign = get_staff_num_and_clef(clef_name)

                    # Sort notes by bbox_x and group by chord threshold
                    note_infos = []
                    for note in clef_notes:
                        if all(k in note for k in ('step', 'octave', 'bbox')) and isinstance(note['bbox'], list):
                            note_infos.append({
                                'step': note['step'],
                                'octave': note['octave'],
                                'bbox_x': float(note['bbox'][0]),
                                'score': note.get('score', 0.0)
                            })
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

                    # Emit notes
                    for chord in chords:
                        for i, note in enumerate(chord):
                            musicxml_content += f'''      <note>
'''
                            if i > 0:
                                musicxml_content += f'''        <chord/>
'''
                            musicxml_content += f'''        <pitch>
          <step>{note['step']}</step>
          <octave>{note['octave']}</octave>
        </pitch>
        <duration>20160</duration>
        <voice>{1 if staff_num == 1 else 2}</voice>
        <type>half</type>
        <staff>{staff_num}</staff>
      </note>
'''

                if not any(clef_dict.values()):
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
                measure_counter += 1

    musicxml_content += '''  </part>
</score-partwise>'''

    # Save file
    try:
        with open(musicxml_filename, 'w', encoding='utf-8') as f:
            f.write(musicxml_content)
        print(f"✓ MusicXML saved to: {musicxml_filename}")
    except Exception as e:
        print(f"✗ Error saving MusicXML: {e}")

    # Display
    try:
        if is_display:
            env = environment.Environment()
            env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
            score = converter.parse(musicxml_filename)
            score.show()
    except Exception as e:
        print(f"✗ Error displaying score: {e}")

    # MIDI
    try:
        if is_midi:
            score = converter.parse(musicxml_filename)
            score.write('midi', fp=midi_filename)
            print(f"🎶 MIDI exported to: {midi_filename}")
    except Exception as e:
        print(f"✗ Error saving MIDI: {e}")

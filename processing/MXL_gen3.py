# import re
# from typing import Dict, Any
# from music21 import converter, environment
# import os
# from pathlib import Path
# from datetime import datetime
#
# def create_exact_musicxml_from_nested_results(
#     nested_results: Dict[str, Dict[str, Dict[str, Dict[str, list]]]],
#     pdf_path: str = None,
#     output_dir: str = "results",
#     is_display: bool = True,
#     is_midi: bool = True,
#     chord_threshold: float = 30.0
# ):
#     def get_staff_num_and_clef(clef_name: str) -> tuple:
#         if clef_name.lower() == 'gclef':
#             return 1, 'G'
#         elif clef_name.lower() == 'fclef':
#             return 2, 'F'
#         else:
#             return 1, 'G'  # Default
#
#     def get_line_for_clef(clef_sign: str) -> int:
#         return 2 if clef_sign == 'G' else 4
#
#     # === Setup Output Path ===
#     if pdf_path:
#         pdf_name = Path(pdf_path).stem
#     else:
#         first_key = next(iter(nested_results))
#         pdf_match = re.match(r'([^_]+(?:_[^_]+)*-\d+)_page_\d+_', first_key)
#         pdf_name = pdf_match.group(1) if pdf_match else "unknown_pdf"
#
#     save_dir = os.path.join(output_dir, pdf_name)
#     os.makedirs(save_dir, exist_ok=True)
#
#     musicxml_filename = os.path.join(save_dir, f"{pdf_name}_results.musicxml")
#     midi_filename = os.path.join(save_dir, f"{pdf_name}_results.mid")
#
#     now = datetime.now().strftime("%Y-%m-%d")
#
#     # === Begin MusicXML content ===
#     musicxml_content = f'''<?xml version="1.0" encoding="utf-8"?>
# <!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
# <score-partwise version="4.0">
#   <work>
#     <work-title>Generated Score</work-title>
#   </work>
#   <identification>
#     <creator type="composer">Sheet2Music</creator>
#     <encoding>
#       <encoding-date>{now}</encoding-date>
#       <software>Python</software>
#     </encoding>
#   </identification>
#   <defaults>
#     <scaling>
#       <millimeters>7</millimeters>
#       <tenths>40</tenths>
#     </scaling>
#   </defaults>
#   <part-list>
#     <score-part id="P1">
#       <part-name>Piano</part-name>
#     </score-part>
#   </part-list>
#   <part id="P1">
# '''
#
#     measure_counter = 1
#     for page_id in sorted(nested_results.keys()):
#         page_data = nested_results[page_id]
#         for group_id in sorted(page_data.keys()):
#             group_data = page_data[group_id]
#             for measure_id in sorted(group_data.keys(), key=lambda x: int(re.search(r'\d+', x).group())):
#                 clef_dict = group_data[measure_id]
#
#                 musicxml_content += f'''    <measure number="{measure_counter}">
# '''
#                 if measure_counter == 1:
#                     musicxml_content += f'''      <attributes>
#         <divisions>10080</divisions>
#         <key>
#           <fifths>0</fifths>
#         </key>
#         <time>
#           <beats>4</beats>
#           <beat-type>4</beat-type>
#         </time>
#         <staves>2</staves>
# '''
#                     clef_done = set()
#                     for clef_name in clef_dict:
#                         staff_num, clef_sign = get_staff_num_and_clef(clef_name)
#                         if staff_num not in clef_done:
#                             musicxml_content += f'''        <clef number="{staff_num}">
#           <sign>{clef_sign}</sign>
#           <line>{get_line_for_clef(clef_sign)}</line>
#         </clef>
# '''
#                             clef_done.add(staff_num)
#                     musicxml_content += f'''      </attributes>
# '''
#
#                 for clef_name, clef_notes in clef_dict.items():
#                     staff_num, _clef_sign = get_staff_num_and_clef(clef_name)
#
#                     # Sort notes by bbox_x and group by chord threshold
#                     note_infos = []
#                     for note in clef_notes:
#                         if all(k in note for k in ('step', 'octave', 'bbox')) and isinstance(note['bbox'], list):
#                             note_infos.append({
#                                 'step': note['step'],
#                                 'octave': note['octave'],
#                                 'bbox_x': float(note['bbox'][0]),
#                                 'score': note.get('score', 0.0)
#                             })
#                     note_infos.sort(key=lambda x: x['bbox_x'])
#
#                     chords = []
#                     current_chord = []
#                     for note_info in note_infos:
#                         if not current_chord:
#                             current_chord.append(note_info)
#                         else:
#                             last_x = current_chord[-1]['bbox_x']
#                             current_x = note_info['bbox_x']
#                             if current_x - last_x <= chord_threshold:
#                                 current_chord.append(note_info)
#                             else:
#                                 chords.append(current_chord)
#                                 current_chord = [note_info]
#                     if current_chord:
#                         chords.append(current_chord)
#
#                     # Emit notes
#                     for chord in chords:
#                         for i, note in enumerate(chord):
#                             musicxml_content += f'''      <note>
# '''
#                             if i > 0:
#                                 musicxml_content += f'''        <chord/>
# '''
#                             musicxml_content += f'''        <pitch>
#           <step>{note['step']}</step>
#           <octave>{note['octave']}</octave>
#         </pitch>
#         <duration>20160</duration>
#         <voice>{1 if staff_num == 1 else 2}</voice>
#         <type>half</type>
#         <staff>{staff_num}</staff>
#       </note>
# '''
#
#                 if not any(clef_dict.values()):
#                     musicxml_content += '''      <note>
#         <rest/>
#         <duration>40320</duration>
#         <voice>1</voice>
#         <type>whole</type>
#         <staff>1</staff>
#       </note>
# '''
#                 musicxml_content += '''    </measure>
# '''
#                 measure_counter += 1
#
#     musicxml_content += '''  </part>
# </score-partwise>'''
#
#     # Save file
#     try:
#         with open(musicxml_filename, 'w', encoding='utf-8') as f:
#             f.write(musicxml_content)
#         print(f"✓ MusicXML saved to: {musicxml_filename}")
#     except Exception as e:
#         print(f"✗ Error saving MusicXML: {e}")
#
#     # Display
#     try:
#         if is_display:
#             env = environment.Environment()
#             env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
#             score = converter.parse(musicxml_filename)
#             score.show()
#     except Exception as e:
#         print(f"✗ Error displaying score: {e}")
#
#     # MIDI
#     try:
#         if is_midi:
#             score = converter.parse(musicxml_filename)
#             score.write('midi', fp=midi_filename)
#             print(f"🎶 MIDI exported to: {midi_filename}")
#     except Exception as e:
#         print(f"✗ Error saving MIDI: {e}")
# ==============================================================================
# import re
# from typing import Dict, Any
# from music21 import converter, environment
# import os
# from pathlib import Path
# from datetime import datetime
#
# def create_exact_musicxml_from_nested_results(
#     nested_results: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]],
#     pdf_path: str = None,
#     output_dir: str = "results",
#     is_display: bool = True,
#     is_midi: bool = True,
#     chord_threshold: float = 30.0
# ):
#     def get_line_for_clef(clef_sign: str) -> int:
#         return 2 if clef_sign == 'G' else 4
#
#     if pdf_path:
#         pdf_name = Path(pdf_path).stem
#     else:
#         first_key = next(iter(nested_results))
#         pdf_match = re.match(r'([^_]+(?:_[^_]+)*-\d+)_page_\d+_', first_key)
#         pdf_name = pdf_match.group(1) if pdf_match else "unknown_pdf"
#
#     save_dir = os.path.join(output_dir, pdf_name)
#     os.makedirs(save_dir, exist_ok=True)
#
#     musicxml_filename = os.path.join(save_dir, f"{pdf_name}_results.musicxml")
#     midi_filename = os.path.join(save_dir, f"{pdf_name}_results.mid")
#
#     now = datetime.now().strftime("%Y-%m-%d")
#
#     musicxml_content = f'''<?xml version="1.0" encoding="utf-8"?>
# <!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
# <score-partwise version="4.0">
#   <work>
#     <work-title>Generated Score</work-title>
#   </work>
#   <identification>
#     <creator type="composer">Sheet2Music</creator>
#     <encoding>
#       <encoding-date>{now}</encoding-date>
#       <software>Python</software>
#     </encoding>
#   </identification>
#   <defaults>
#     <scaling>
#       <millimeters>7</millimeters>
#       <tenths>40</tenths>
#     </scaling>
#   </defaults>
#   <part-list>
#     <score-part id="P1">
#       <part-name>Piano</part-name>
#     </score-part>
#   </part-list>
#   <part id="P1">
# '''
#
#     measure_counter = 1
#
#     for page_id in sorted(nested_results.keys()):
#         page_data = nested_results[page_id]
#
#         for group_index, (group_id, group_measures) in enumerate(sorted(page_data.items())):
#             is_first_group = group_index == 0
#
#             for measure_index, (measure_id, clef_dict) in enumerate(sorted(group_measures.items(), key=lambda x: int(re.search(r'\d+', x[0]).group()))):
#                 musicxml_content += f'    <measure number="{measure_counter}">\n'
#
#                 # Force new line for each group (except first)
#                 if measure_index == 0 and not is_first_group:
#                     musicxml_content += '      <print new-system="yes"/>\n'
#
#                 # Add attributes only on first measure or start of new group
#                 if measure_counter == 1 or measure_index == 0:
#                     musicxml_content += (
#                         '      <attributes>\n'
#                         '        <divisions>10080</divisions>\n'
#                         '        <key>\n'
#                         '          <fifths>0</fifths>\n'
#                         '        </key>\n'
#                         '        <time>\n'
#                         '          <beats>4</beats>\n'
#                         '          <beat-type>4</beat-type>\n'
#                         '        </time>\n'
#                         '        <staves>2</staves>\n'
#                     )
#                     clef_done = set()
#                     for clef_idx in sorted(clef_dict.keys()):
#                         clef_type = clef_dict[clef_idx].get('clef_type', 'gClef')
#                         staff_num = 1 if 'g' in clef_type.lower() else 2
#                         if staff_num not in clef_done:
#                             clef_sign = 'G' if clef_type.lower() == 'gclef' else 'F'
#                             musicxml_content += (
#                                 f'        <clef number="{staff_num}">\n'
#                                 f'          <sign>{clef_sign}</sign>\n'
#                                 f'          <line>{get_line_for_clef(clef_sign)}</line>\n'
#                                 f'        </clef>\n'
#                             )
#                             clef_done.add(staff_num)
#                     musicxml_content += '      </attributes>\n'
#
#                 all_notes = []
#                 for clef_index, clef_data in clef_dict.items():
#                     clef_type = clef_data.get('clef_type', 'gClef')
#                     staff_num = 1 if 'g' in clef_type.lower() else 2
#
#                     for note in clef_data.get('notes', []):
#                         if all(k in note for k in ('step', 'octave', 'bbox')) and isinstance(note['bbox'], list):
#                             all_notes.append({
#                                 'step': note['step'],
#                                 'octave': note['octave'],
#                                 'bbox_x': float(note['bbox'][0]),
#                                 'staff': staff_num,
#                             })
#
#                 all_notes.sort(key=lambda x: x['bbox_x'])
#
#                 aligned_groups = []
#                 current_group = []
#
#                 for note in all_notes:
#                     if not current_group:
#                         current_group.append(note)
#                     else:
#                         avg_x = sum(n['bbox_x'] for n in current_group) / len(current_group)
#                         if abs(note['bbox_x'] - avg_x) <= chord_threshold:
#                             current_group.append(note)
#                         else:
#                             aligned_groups.append(current_group)
#                             current_group = [note]
#                 if current_group:
#                     aligned_groups.append(current_group)
#
#                 duration_val = 20160
#
#                 for group in aligned_groups:
#                     notes_by_staff = {}
#                     for note in group:
#                         notes_by_staff.setdefault(note['staff'], []).append(note)
#                     staffs_in_group = sorted(notes_by_staff.keys())
#                     for i, staff in enumerate(staffs_in_group):
#                         notes = notes_by_staff[staff]
#                         notes.sort(key=lambda n: n['bbox_x'])
#                         for note_idx, note in enumerate(notes):
#                             musicxml_content += '      <note>\n'
#                             if note_idx > 0:
#                                 musicxml_content += '        <chord/>\n'
#                             musicxml_content += (
#                                 f'        <pitch>\n'
#                                 f'          <step>{note["step"]}</step>\n'
#                                 f'          <octave>{note["octave"]}</octave>\n'
#                                 f'        </pitch>\n'
#                                 f'        <duration>{duration_val}</duration>\n'
#                                 f'        <voice>{staff}</voice>\n'
#                                 f'        <type>half</type>\n'
#                                 f'        <staff>{staff}</staff>\n'
#                                 f'      </note>\n'
#                             )
#                         if i < len(staffs_in_group) - 1:
#                             musicxml_content += (
#                                 f'      <backup>\n'
#                                 f'        <duration>{duration_val}</duration>\n'
#                                 f'      </backup>\n'
#                             )
#
#                 if not all_notes:
#                     musicxml_content += (
#                         '      <note>\n'
#                         '        <rest/>\n'
#                         f'        <duration>{duration_val * 2}</duration>\n'
#                         '        <voice>1</voice>\n'
#                         '        <type>whole</type>\n'
#                         '        <staff>1</staff>\n'
#                         '      </note>\n'
#                     )
#
#                 musicxml_content += '    </measure>\n'
#                 measure_counter += 1
#
#     musicxml_content += '  </part>\n</score-partwise>'
#
#     try:
#         with open(musicxml_filename, 'w', encoding='utf-8') as f:
#             f.write(musicxml_content)
#         print(f"✓ MusicXML saved to: {musicxml_filename}")
#     except Exception as e:
#         print(f"✗ Error saving MusicXML: {e}")
#
#     try:
#         if is_display:
#             env = environment.Environment()
#             env['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
#             score = converter.parse(musicxml_filename)
#             score.show()
#     except Exception as e:
#         print(f"✗ Error displaying score: {e}")
#
#     try:
#         if is_midi:
#             score = converter.parse(musicxml_filename)
#             score.write('midi', fp=midi_filename)
#             print(f"🎶 MIDI exported to: {midi_filename}")
#     except Exception as e:
#         print(f"✗ Error saving MIDI: {e}")
# =====================================================================
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from music21 import converter, environment


class MusicXMLGenerator:
    def __init__(self,
                 output_dir: str = "results",
                 is_display: bool = True,
                 is_midi: bool = True,
                 chord_threshold: float = 30.0,
                 musicxml_path: str = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"):
        self.output_dir = output_dir
        self.is_display = is_display
        self.is_midi = is_midi
        self.chord_threshold = chord_threshold
        self.musicxml_path = musicxml_path

    def _get_line_for_clef(self, clef_sign: str) -> int:
        return 2 if clef_sign == 'G' else 4

    def _group_chords(self, notes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        notes.sort(key=lambda x: x['bbox_x'])
        groups = []
        current_group = []

        for note in notes:
            if not current_group:
                current_group.append(note)
            else:
                avg_x = sum(n['bbox_x'] for n in current_group) / len(current_group)
                if abs(note['bbox_x'] - avg_x) <= self.chord_threshold:
                    current_group.append(note)
                else:
                    groups.append(current_group)
                    current_group = [note]
        if current_group:
            groups.append(current_group)

        return groups

    def generate(self, nested_results: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]],
                 pdf_path: str = None):
        # Determine file names
        if pdf_path:
            pdf_name = Path(pdf_path).stem
        else:
            first_key = next(iter(nested_results))
            match = re.match(r'([^_]+(?:_[^_]+)*-\d+)_page_\d+_', first_key)
            pdf_name = match.group(1) if match else "unknown_pdf"

        save_dir = os.path.join(self.output_dir, pdf_name)
        os.makedirs(save_dir, exist_ok=True)

        musicxml_file = os.path.join(save_dir, f"{pdf_name}_results.musicxml")
        midi_file = os.path.join(save_dir, f"{pdf_name}_results.mid")

        now = datetime.now().strftime("%Y-%m-%d")
        xml = []

        # MusicXML Header
        xml.append('<?xml version="1.0" encoding="utf-8"?>')
        xml.append('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
                   '"http://www.musicxml.org/dtds/partwise.dtd">')
        xml.append('<score-partwise version="4.0">')
        xml.append(f'''  <work>
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
  <part id="P1">''')

        measure_counter = 1
        for page_id in sorted(nested_results.keys()):
            page_data = nested_results[page_id]
            for group_index, (group_id, measures) in enumerate(sorted(page_data.items())):
                is_first_group = group_index == 0

                for measure_index, (measure_id, clefs) in enumerate(
                        sorted(measures.items(), key=lambda x: int(re.search(r'\d+', x[0]).group()))):
                    xml.append(f'    <measure number="{measure_counter}">')

                    # Line break for each new group (except first)
                    if measure_index == 0 and not is_first_group:
                        xml.append('      <print new-system="yes"/>')

                    # Add header on first or new group measure
                    if measure_counter == 1 or measure_index == 0:
                        xml.append('      <attributes>')
                        xml.append('        <divisions>10080</divisions>')
                        xml.append('        <key><fifths>0</fifths></key>')
                        xml.append('        <time><beats>4</beats><beat-type>4</beat-type></time>')
                        xml.append('        <staves>2</staves>')

                        seen_staffs = set()
                        for idx in sorted(clefs.keys()):
                            clef_type = clefs[idx].get("clef_type", "gClef").lower()
                            staff = 1 if "g" in clef_type else 2
                            if staff not in seen_staffs:
                                sign = "G" if staff == 1 else "F"
                                line = self._get_line_for_clef(sign)
                                xml.append(f'        <clef number="{staff}"><sign>{sign}</sign><line>{line}</line></clef>')
                                seen_staffs.add(staff)
                        xml.append('      </attributes>')

                    all_notes = []
                    for idx, data in clefs.items():
                        staff = 1 if 'g' in data.get("clef_type", "gClef").lower() else 2
                        for note in data.get("notes", []):
                            if {'step', 'octave', 'bbox'} <= note.keys():
                                all_notes.append({
                                    'step': note['step'],
                                    'octave': note['octave'],
                                    'bbox_x': float(note['bbox'][0]),
                                    'staff': staff
                                })

                    duration_val = 20160
                    aligned_groups = self._group_chords(all_notes)

                    for group in aligned_groups:
                        notes_by_staff = {}
                        for note in group:
                            notes_by_staff.setdefault(note['staff'], []).append(note)

                        staffs = sorted(notes_by_staff.keys())
                        for i, staff in enumerate(staffs):
                            notes = notes_by_staff[staff]
                            notes.sort(key=lambda n: n['bbox_x'])
                            for j, note in enumerate(notes):
                                xml.append('      <note>')
                                if j > 0:
                                    xml.append('        <chord/>')
                                xml.append(f'        <pitch><step>{note["step"]}</step><octave>{note["octave"]}</octave></pitch>')
                                xml.append(f'        <duration>{duration_val}</duration><voice>{staff}</voice><type>half</type><staff>{staff}</staff>')
                                xml.append('      </note>')
                            if i < len(staffs) - 1:
                                xml.append(f'      <backup><duration>{duration_val}</duration></backup>')

                    if not all_notes:
                        xml.append('      <note><rest/><duration>{}</duration><voice>1</voice><type>whole</type><staff>1</staff></note>'.format(duration_val * 2))

                    xml.append('    </measure>')
                    measure_counter += 1

        xml.append('  </part>\n</score-partwise>')

        try:
            with open(musicxml_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(xml))
            print(f"✓ MusicXML saved to: {musicxml_file}")
        except Exception as e:
            print(f"✗ Error saving MusicXML: {e}")

        if self.is_display:
            try:
                env = environment.Environment()
                env['musicxmlPath'] = self.musicxml_path
                score = converter.parse(musicxml_file)
                score.show()
            except Exception as e:
                print(f"✗ Error displaying score: {e}")

        if self.is_midi:
            try:
                score = converter.parse(musicxml_file)
                score.write('midi', fp=midi_file)
                print(f"🎶 MIDI exported to: {midi_file}")
            except Exception as e:
                print(f"✗ Error saving MIDI: {e}")




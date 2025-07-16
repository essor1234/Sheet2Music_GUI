import streamlit as st
from pathlib import Path
from main import pipe_line
import time

# Configure Streamlit page
st.set_page_config(page_title="Sheet2Music", layout="centered")
st.title("🎼 Sheet2Music - Convert PDF Sheet Music to MusicXML & MIDI")

# Initialize session state
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "pdf_stem" not in st.session_state:
    st.session_state.pdf_stem = ""

uploaded_file = st.file_uploader("Upload your sheet music (PDF)", type=["pdf"])

if uploaded_file is not None:
    filename = uploaded_file.name
    pdf_stem = Path(filename).stem
    save_path = Path("input_pdfs") / filename
    save_path.parent.mkdir(exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File '{filename}' uploaded successfully.")

    if st.button("Start Processing 🎶"):
        with st.spinner("Running pipeline... Please wait..."):
            progress = st.progress(0)
            for pct in range(0, 100, 10):
                time.sleep(0.2)
                progress.progress(pct)

            pipe_line(str(save_path))
            progress.progress(100)

        st.session_state.processing_done = True
        st.session_state.pdf_stem = pdf_stem
        st.success("✅ Processing completed!")

# Output section
if st.session_state.processing_done:
    pdf_stem = st.session_state.pdf_stem
    result_folder = Path("data_storage/results_path") / pdf_stem
    xml_path = result_folder / f"{pdf_stem}_results.musicxml"
    midi_path = result_folder / f"{pdf_stem}_results.mid"

    st.markdown("### ✅ Output Files")

    if xml_path.exists():
        st.markdown("#### 🎵 MusicXML")
        with open(xml_path, "rb") as f:
            st.download_button(
                label="📥 Download MusicXML",
                data=f,
                file_name=xml_path.name,
                mime="application/xml"
            )
    else:
        st.warning("❌ MusicXML file not found.")

    if midi_path.exists():
        st.markdown("#### 🎹 MIDI")
        with open(midi_path, "rb") as f:
            midi_bytes = f.read()
            st.download_button(
                label="📥 Download MIDI",
                data=midi_bytes,
                file_name=midi_path.name,
                mime="audio/midi"
            )
    else:
        st.warning("❌ MIDI file not found.")

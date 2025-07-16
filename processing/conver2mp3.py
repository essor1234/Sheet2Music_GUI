import subprocess
from pathlib import Path
from pydub import AudioSegment

def midi_to_mp3(midi_path: Path, sf2_path: Path, mp3_path: Path, fluidsynth_exe):
    wav_path = midi_path.with_suffix(".wav")

    if not fluidsynth_exe.exists():
        print("❌ fluidsynth.exe not found!")
        return

    if not sf2_path.exists():
        print("❌ SoundFont file not found!")
        return

    print(f"🎵 Running FluidSynth to convert MIDI to WAV...")
    cmd = [
        str(fluidsynth_exe.resolve()),
        "-ni",
        str(sf2_path.resolve()),
        str(midi_path.resolve()),
        "-F", str(wav_path.resolve()),
        "-r", "44100"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ WAV created: {wav_path}")
    except subprocess.CalledProcessError as e:
        print("❌ FluidSynth failed:", e)
        return

    if wav_path.exists():
        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3")
        wav_path.unlink()
        print(f"✅ MP3 created: {mp3_path}")
    else:
        print("❌ WAV was not created.")

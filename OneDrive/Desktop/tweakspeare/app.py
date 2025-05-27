import streamlit as st
import whisper
import pyaudio
import wave
import time
import os

# Load Whisper model
try:
    model = whisper.load_model("tiny")
except Exception as e:
    st.error(f"Failed to load Whisper model: {str(e)}")
    st.stop()

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
OUTPUT_WAV = "temp_audio.wav"

# Function to list available microphones
def list_mics():
    p = pyaudio.PyAudio()
    mics = []
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            mics.append((i, device_info.get('name')))
    p.terminate()
    return mics

# Function to record audio
def record_audio(audio_file, device_index=None):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                           frames_per_buffer=CHUNK, input_device_index=device_index)
        frames = []
        start_time = time.time()
        while st.session_state.recording and time.time() - start_time < 30:  # Limit to 30s
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if time.time() - start_time > 1:
                st.session_state.placeholder.text(f"Recording... ({len(frames)} frames)")
                start_time = time.time()
        stream.stop_stream()
        stream.close()
        audio.terminate()
        if frames:
            wf = wave.open(audio_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            st.session_state.placeholder.text("Audio recorded successfully.")
        else:
            st.session_state.transcribed_text = "Error: No audio data recorded."
            st.session_state.placeholder.text("")
    except Exception as e:
        st.session_state.transcribed_text = f"Recording error: {str(e)}"
        st.session_state.placeholder.text("")

# Function to transcribe audio
def transcribe_audio(audio_file):
    try:
        result = model.transcribe(audio_file, verbose=False)
        return result["text"]
    except Exception as e:
        return f"Transcription error: {str(e)}"

# Set up the app
st.title("Tweakspeare: Improve Your English Vocabulary")
st.write("Speak and see your transcription!")

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'placeholder' not in st.session_state:
    st.session_state.placeholder = st.empty()

# Microphone selection
mics = list_mics()
st.sidebar.header("Microphones")
if mics:
    mic_names = [name for _, name in mics]
    selected_mic = st.sidebar.selectbox("Choose mic", mic_names, key="mic_select")
    device_index = next(i for i, name in mics if name == selected_mic)
    st.sidebar.write(f"Selected: {selected_mic}")
else:
    st.sidebar.error("No microphones detected!")
    device_index = None
    st.session_state.transcribed_text = "Error: No microphones available."
    st.stop()

# Single Record button
if st.button("Record", key="record"):
    st.session_state.recording = not st.session_state.recording
    if st.session_state.recording:
        st.session_state.transcribed_text = ""
        st.session_state.placeholder.text("Recording...")
        record_audio(OUTPUT_WAV, device_index)
    else:
        st.session_state.placeholder.text("Processing transcription...")
        try:
            if os.path.exists(OUTPUT_WAV):
                transcription = transcribe_audio(OUTPUT_WAV)
                st.session_state.transcribed_text = transcription
                st.session_state.placeholder.text("")
                os.remove(OUTPUT_WAV)
            else:
                st.session_state.transcribed_text = "Error: No audio file recorded."
                st.session_state.placeholder.text("")
        except Exception as e:
            st.session_state.transcribed_text = f"Error: {str(e)}"
            st.session_state.placeholder.text("")

# Display transcription
if st.session_state.transcribed_text:
    st.subheader("Transcribed Text")
    st.write(st.session_state.transcribed_text)

# Sidebar instructions
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Select your wired earphone mic.
2. Click 'Record' to start.
3. Click 'Record' again to stop.
4. See transcription.
**Note**: Offline with Whisper.
""")
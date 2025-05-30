import streamlit as st
import whisper
import pyaudio
import wave
import os
import time
import threading
import tempfile
import re
import uuid
from pathlib import Path
from gtts import gTTS
import pygame
import requests
from dotenv import load_dotenv
import json

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Temp directory
TEMP_DIR = Path(tempfile.gettempdir()) / "tweakspeare"
TEMP_DIR.mkdir(exist_ok=True)

# Load environment variables
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Load Whisper model
@st.cache_resource
def load_model():
    try:
        model = whisper.load_model("medium")  # Medium for precise transcription
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        return None

# Check errors using xAI Grok API
def check_errors_grok(text):
    if not XAI_API_KEY:
        st.error("xAI API key not found. Set XAI_API_KEY in .env file.")
        return []
    
    try:
        url = "https://api.grok.xai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = (
            f"Analyze the following text for grammar and vocabulary errors with 100% precision. "
            f"For each error, provide: "
            f"1) incorrect word/phrase (exact match), "
            f"2) explanation of the error, "
            f"3) one correct suggestion, "
            f"4) error type (spelling, grammar, vocabulary), "
            f"5) severity (high). "
            f"Return results as a JSON list of objects with fields: "
            f"'error', 'message', 'suggestions' (list with one item), 'type', 'severity'. "
            f"Locate errors by word position for start/end indices. "
            f"Example: For 'I writting a books', detect 'writting' (spelling), 'books' (grammar), 'is' (grammar). "
            f"Text: '{text}'"
        )
        payload = {
            "model": "grok-beta",
            "messages": [
                {"role": "system", "content": "You are a precise English grammar and vocabulary expert."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.0
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        errors = []
        try:
            content = result["choices"][0]["message"]["content"]
            parsed_errors = json.loads(content)
            for err in parsed_errors:
                start = text.find(err["error"])
                if start == -1:
                    continue
                errors.append({
                    "error": err["error"],
                    "message": err["message"],
                    "suggestions": err["suggestions"],
                    "start": start,
                    "end": start + len(err["error"]),
                    "type": err["type"],
                    "severity": err["severity"]
                })
        except (KeyError, json.JSONDecodeError) as e:
            st.error(f"Error parsing Grok response: {str(e)}")
            return []
        
        return errors
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to xAI API: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error processing grammar check: {str(e)}")
        return []

# Check dependencies
def check_dependencies():
    missing_deps = []
    for module, name in [
        ('whisper', 'openai-whisper'),
        ('pyaudio', 'pyaudio'),
        ('gtts', 'gTTS'),
        ('pygame', 'pygame'),
        ('requests', 'requests'),
        ('dotenv', 'python-dotenv')
    ]:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(name)
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}")
        return False
    return True

# List microphones
@st.cache_data
def list_microphones():
    try:
        p = pyaudio.PyAudio()
        microphones = []
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    name = device_info.get('name', f'Device {i}')
                    microphones.append((i, name))
            except:
                continue
        p.terminate()
        return microphones
    except Exception as e:
        st.error(f"Error accessing audio devices: {str(e)}")
        return []

class AudioRecorder:
    def __init__(self, device_index=None):
        self.device_index = device_index
        self.is_recording = False
        self.audio_frames = []
        self.recording_thread = None
        self.temp_file = None
        self.error_message = None
    
    def start_recording(self):
        if self.is_recording:
            return False
        try:
            self.is_recording = True
            self.audio_frames = []
            self.error_message = None
            self.recording_thread = threading.Thread(target=self._record_audio_thread)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            return True
        except Exception as e:
            self.error_message = f"Failed to start recording: {str(e)}"
            self.is_recording = False
            return False
    
    def stop_recording(self):
        if not self.is_recording:
            return None
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5.0)
        if self.error_message:
            return None, self.error_message
        return self._save_audio_file(), None
    
    def _record_audio_thread(self):
        audio_interface = None
        stream = None
        try:
            audio_interface = pyaudio.PyAudio()
            try:
                device_info = audio_interface.get_device_info_by_index(self.device_index)
                if device_info.get('maxInputChannels', 0) == 0:
                    self.error_message = f"Device {self.device_index} has no input channels"
                    return
            except Exception as e:
                self.error_message = f"Invalid device index {self.device_index}: {str(e)}"
                return
            stream = audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )
            stream.start_stream()
            while self.is_recording:
                try:
                    audio_data = stream.read(CHUNK, exception_on_overflow=False)
                    if audio_data:
                        self.audio_frames.append(audio_data)
                    time.sleep(0.001)
                except Exception as e:
                    self.error_message = f"Error during recording: {str(e)}"
                    break
        except Exception as e:
            self.error_message = f"Recording setup error: {str(e)}"
        finally:
            try:
                if stream and stream.is_active():
                    stream.stop_stream()
                if stream:
                    stream.close()
            except:
                pass
            try:
                if audio_interface:
                    audio_interface.terminate()
            except:
                pass
    
    def _save_audio_file(self):
        if not self.audio_frames:
            return None
        try:
            temp_filename = f"recording_{uuid.uuid4().hex}.wav"
            temp_path = TEMP_DIR / temp_filename
            p = pyaudio.PyAudio()
            sample_width = p.get_sample_size(FORMAT)
            p.terminate()
            with wave.open(str(temp_path), 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(RATE)
                wav_file.writeframes(b''.join(self.audio_frames))
            if temp_path.exists() and temp_path.stat().st_size > 44:
                self.temp_file = str(temp_path)
                return str(temp_path)
            return None
        except Exception as e:
            self.error_message = f"Error saving audio: {str(e)}"
            return None
    
    def cleanup(self):
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                self.temp_file = None
            except:
                pass

def post_process_transcription(text, custom_corrections=None):
    tweakspeare_variants = [
        "tweak spear", "tweak sphere", "tweak spare", "tweak spur",
        "tweet spear", "tweet sphere", "tweet spare", "tweak spire",
        "weak spear", "weak sphere", "weak spare", "twig spear",
        "tweak shakespeare", "tweet shakespeare", "tweek spear",
        "tweak pier", "tweak beer", "tweak peer", "tweak sheer",
        "tweaks peer", "tweaks spear", "tweaksphere", "tweak spiers",
        "tweak spears", "tweek sphere", "tweak sphear", "tweak spehere"
    ]
    corrected_text = text
    for variant in tweakspeare_variants:
        pattern = re.compile(re.escape(variant), re.IGNORECASE)
        corrected_text = pattern.sub("Tweakspeare", corrected_text)
    if custom_corrections:
        for wrong, correct in custom_corrections.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            corrected_text = pattern.sub(correct, corrected_text)
    return corrected_text

def transcribe_audio_file(model, audio_file_path):
    try:
        if not os.path.exists(audio_file_path):
            return "Error: Audio file not found"
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            return "Error: Audio recording too short or empty"
        result = model.transcribe(
            audio_file_path,
            language='en',
            verbose=False,
            fp16=False,
            temperature=0.0,
            suppress_tokens=[],
            condition_on_previous_text=False
        )
        transcribed_text = result["text"].strip()
        if not transcribed_text:
            return "No speech detected in the recording"
        corrected_text = post_process_transcription(transcribed_text)
        return corrected_text
    except Exception as e:
        return f"Transcription error: {str(e)}"

def reframe_text_with_corrections(original_text, errors):
    if not errors:
        return original_text
    corrected_text = original_text
    sorted_errors = sorted(errors, key=lambda x: x['start'], reverse=True)
    for error in sorted_errors:
        if error['suggestions']:
            start, end = error['start'], error['end']
            corrected_text = corrected_text[:start] + error['suggestions'][0] + corrected_text[end:]
    return corrected_text

def generate_pronunciation(text):
    try:
        pronunciation_file = TEMP_DIR / f"pronunciation_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(str(pronunciation_file))
        return str(pronunciation_file)
    except Exception as e:
        st.error(f"Error generating pronunciation: {str(e)}")
        return None

def play_pronunciation(audio_file):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.unload()
        os.remove(audio_file)
    except Exception as e:
        st.error(f"Error playing pronunciation: {str(e)}")

def display_highlighted_text(text, errors):
    if not errors:
        return text
    html_text = text
    offset = 0
    sorted_errors = sorted(errors, key=lambda x: x['start'])
    for error in sorted_errors:
        start = error['start'] + offset
        end = error['end'] + offset
        error_text = error['error']
        suggestion = error['suggestions'][0] if error['suggestions'] else error_text
        highlighted = f'<span style="background-color: #e74c3c; padding: 2px; color: white; font-weight: bold;" title="{suggestion}">{error_text}</span>'
        html_text = html_text[:start] + highlighted + html_text[end:]
        offset += len(highlighted) - (end - start)
    return html_text

def initialize_session_state():
    default_values = {
        "recorder": None,
        "transcription_result": "",
        "corrected_text": "",
        "is_currently_recording": False,
        "selected_mic_index": None,
        "recording_start_time": None,
        "transcription_history": [],
        "custom_corrections": {}
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def format_duration(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def update_timer():
    if st.session_state.is_currently_recording and st.session_state.recording_start_time:
        elapsed = time.time() - st.session_state.recording_start_time
        return format_duration(elapsed)
    return "00:00"

def main():
    st.set_page_config(
        page_title="Tweakspeare - Speech to Text",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    if not check_dependencies():
        st.stop()
    model = load_model()
    if model is None:
        st.stop()
    st.title("🎙️ Tweakspeare: AI-Powered Speech Transcription")
    st.markdown("**Improve your spoken English with precise transcription and corrections**")
    initialize_session_state()
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. Select your microphone
        2. Click 'Start Recording'
        3. Speak naturally
        4. Click 'Stop Recording'
        5. Review transcription and corrections
        """)
        st.subheader("🔧 Custom Corrections")
        if st.checkbox("Enable custom word corrections"):
            col1, col2 = st.columns(2)
            with col1:
                wrong_word = st.text_input("Misheard as:", placeholder="e.g., 'john doe'")
            with col2:
                correct_word = st.text_input("Should be:", placeholder="e.g., 'Jon Doe'")
            if st.button("Add Correction") and wrong_word and correct_word:
                st.session_state.custom_corrections[wrong_word.lower()] = correct_word
                st.success(f"Added: '{wrong_word}' → '{correct_word}'")
            if st.session_state.custom_corrections:
                st.markdown("**Current corrections:**")
                for wrong, correct in st.session_state.custom_corrections.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"'{wrong}' → '{correct}'")
                    with col2:
                        if st.button("❌", key=f"del_{wrong}"):
                            del st.session_state.custom_corrections[wrong]
                            st.rerun()
    microphones = list_microphones()
    if not microphones:
        st.error("❌ No microphones detected!")
        st.stop()
    st.subheader("🎤 Microphone Selection")
    mic_options = [f"{name} (Device {idx})" for idx, name in microphones]
    selected_mic_option = st.selectbox("Choose your microphone:", mic_options)
    selected_device_idx = next(
        idx for idx, name in microphones 
        if f"{name} (Device {idx})" == selected_mic_option
    )
    if st.session_state.selected_mic_index != selected_device_idx:
        st.session_state.selected_mic_index = selected_device_idx
        if st.session_state.recorder:
            st.session_state.recorder.cleanup()
        st.session_state.recorder = AudioRecorder(selected_device_idx)
    st.subheader("Recording Controls")
    control_col1, control_col2, status_col = st.columns([1, 1, 2])
    with control_col1:
        start_disabled = st.session_state.is_currently_recording
        if st.button("Start Recording", disabled=start_disabled, key="start_recording_btn"):
            if st.session_state.recorder:
                success = st.session_state.recorder.start_recording()
                if success:
                    st.session_state.is_currently_recording = True
                    st.session_state.recording_start_time = time.time()
                    st.rerun()
                else:
                    error = getattr(st.session_state.recorder, 'error_message', 'Unknown error')
                    st.error(f"Failed to start recording: {error}")
                    st.session_state.recorder = AudioRecorder(st.session_state.selected_mic_index)
    with control_col2:
        stop_disabled = not st.session_state.is_currently_recording
        if st.button("Stop Recording", disabled=stop_disabled, key="stop_recording_btn"):
            if st.session_state.recorder and st.session_state.is_currently_recording:
                with st.spinner("Processing your audio..."):
                    result = st.session_state.recorder.stop_recording()
                    st.session_state.is_currently_recording = False
                    st.session_state.recording_start_time = None
                    if result[0] is None:
                        error_msg = result[1] if result[1] else "Failed to process audio"
                        st.session_state.transcription_result = f"Error: {error_msg}"
                    else:
                        audio_file_path = result[0]
                        if os.path.exists(audio_file_path) and os.path.getsize(audio_file_path) > 1000:
                            transcription = transcribe_audio_file(model, audio_file_path)
                            st.session_state.transcription_result = transcription
                            if not transcription.startswith("Error:") and not transcription.startswith("No speech"):
                                errors = check_errors_grok(transcription)
                                st.session_state.corrected_text = reframe_text_with_corrections(transcription, errors)
                                st.session_state.transcription_history.append({
                                    "timestamp": time.strftime("%H:%M:%S"),
                                    "original": transcription,
                                    "corrected": st.session_state.corrected_text,
                                    "errors": len(errors)
                                })
                        else:
                            st.session_state.transcription_result = "Error: Audio too short or empty"
                        st.session_state.recorder.cleanup()
                    st.rerun()
    with status_col:
        if st.session_state.is_currently_recording:
            current_time = update_timer()
            st.markdown(f"🔴 **RECORDING... {current_time}**")
            time.sleep(1)
            st.rerun()
        else:
            st.empty()
    if st.session_state.transcription_result:
        if st.session_state.transcription_result.startswith("Error:"):
            st.error(st.session_state.transcription_result)
        elif st.session_state.transcription_result.startswith("No speech"):
            st.warning(st.session_state.transcription_result)
        else:
            tab1, tab2 = st.tabs(["Your Speech", "Corrected Version"])
            with tab1:
                st.subheader("What You Said")
                errors = check_errors_grok(st.session_state.transcription_result)
                if errors:
                    st.info(f"Found {len(errors)} errors")
                    highlighted_html = display_highlighted_text(st.session_state.transcription_result, errors)
                    st.markdown(highlighted_html, unsafe_allow_html=True)
                    st.subheader("Detailed Corrections")
                    for i, error in enumerate(errors):
                        with st.expander(f"Error: '{error['error']}'"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**Issue**: {error['message']}")
                                if error['suggestions']:
                                    st.write(f"**Suggestion**: {error['suggestions'][0]}")
                            with col2:
                                if error['suggestions']:
                                    if st.button(f"Hear '{error['suggestions'][0]}'", key=f"pron_{i}"):
                                        audio_file = generate_pronunciation(error['suggestions'][0])
                                        if audio_file:
                                            play_pronunciation(audio_file)
                else:
                    st.success("No errors detected!")
                    st.text_area("Your speech:", value=st.session_state.transcription_result, height=120)
            with tab2:
                st.subheader("Corrected Version")
                if st.session_state.corrected_text != st.session_state.transcription_result:
                    st.text_area("Corrected text:", value=st.session_state.corrected_text, height=120)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Hear Original"):
                            audio_file = generate_pronunciation(st.session_state.transcription_result)
                            if audio_file:
                                play_pronunciation(audio_file)
                    with col2:
                        if st.button("Hear Corrected"):
                            audio_file = generate_pronunciation(st.session_state.corrected_text)
                            if audio_file:
                                play_pronunciation(audio_file)
                else:
                    st.success("Your speech was perfect!")
                    st.text_area("Your speech:", value=st.session_state.transcription_result, height=120)
    if st.session_state.transcription_history:
        st.subheader("Session History")
        with st.expander(f"View previous recordings ({len(st.session_state.transcription_history)} total)"):
            for i, entry in enumerate(reversed(st.session_state.transcription_history[-5:])):
                st.markdown(f"**{entry['timestamp']}** - {entry['errors']} errors")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"*Original:* {entry['original'][:100]}...")
                with col2:
                    st.markdown(f"*Corrected:* {entry['corrected'][:100]}...")
                st.markdown("---")
        if st.button("Clear History"):
            st.session_state.transcription_history = []
            st.rerun()
    st.markdown("---")
    st.markdown(
        "*Powered by [OpenAI Whisper](https://openai.com/research/whisper) and [xAI Grok](https://x.ai) • Built with [Streamlit](https://streamlit.io)*"
    )

def cleanup_on_exit():
    if hasattr(st.session_state, 'recorder') and st.session_state.recorder:
        st.session_state.recorder.cleanup()
    try:
        for file in TEMP_DIR.glob("*.mp3"):
            file.unlink()
        for file in TEMP_DIR.glob("*.wav"):
            file.unlink()
    except:
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_on_exit()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        cleanup_on_exit()
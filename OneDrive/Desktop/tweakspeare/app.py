import streamlit as st
import speech_recognition as sr

recognizer = sr.Recognizer()

def transcribe_audio():
    with sr.Microphone() as source:
        st.info("Listening... Speak clearly for up to 10 seconds.")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)  # Use Google's API
            return text
        except sr.WaitTimeoutError:
            return "Error: No speech detected."
        except sr.UnknownValueError:
            return "Error: Could not understand the audio."
        except sr.RequestError:
            return "Error: Network issue. Check your internet."
        
st.title("Tweakspeare: Improves Your English Vocabulary")
st.write("Speak, and I'll analyze your grammar, suggest corrections, and score your speech!")

if st.button("Start Recording"):
    st.write("This will start recording your speech (coming soon)!")

st.sidebar.header("How to Use Tweakspeare")
st.sidebar.write("1. Click 'Start Recording' and speak clearly.")                                
st.sidebar.write("2. Stop recording to see your speech analysis.")
st.sidebar.write("3. View errors, corrections, and your score.")
st.sidebar.write("Note: Microphone and internet required.")

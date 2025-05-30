import pyaudio
import wave
import sys
import threading

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048  # Increased buffer
OUTPUT_WAV = "test_output.wav"

def record_audio():
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                           frames_per_buffer=CHUNK, input_device_index=7)  # Try 1 or 14 if fails
        print("Recording... Press Enter to stop.")
        frames = []
        recording = True

        def check_input():
            nonlocal recording
            input()
            recording = False

        input_thread = threading.Thread(target=check_input)
        input_thread.start()

        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()
        if frames:
            wf = wave.open(OUTPUT_WAV, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            print(f"Saved {OUTPUT_WAV}")
        else:
            print("Error: No audio data recorded.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    record_audio()
import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
OUTPUT_WAV = "test_output.wav"

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Recording for 5 seconds...")
frames = []
for _ in range(int(RATE / CHUNK * 5)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)
print("Done recording.")
stream.stop_stream()
stream.close()
audio.terminate()

wf = wave.open(OUTPUT_WAV, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print(f"Saved {OUTPUT_WAV}")
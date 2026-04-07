# record.py — records 4 seconds of audio from microphone
# First install: pip install sounddevice scipy
import sounddevice as sd
import scipy.io.wavfile as wav

SAMPLE_RATE = 16000
DURATION = 4  # seconds

print('Recording in 3... 2... 1... GO!')
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
print('Done! Saved to test_input.wav')
wav.write('test_input.wav', SAMPLE_RATE, audio)

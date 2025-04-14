import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import speech_recognition as sr
from jiwer import cer, wer
from IPython.display import Audio
import whisper
import csv
import os
import tempfile
import wave
from gtts import gTTS

file_path = r'13_speech_recognition/data/speech_01.wav'
audio_signal, sample_rate = librosa.load(file_path, sr=None)
print(sample_rate)

# Plotting an amplitude-time graph to visualize the audio
plt.figure(figsize=(12,4))
librosa.display.waveshow(audio_signal, sr=sample_rate)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
# plt.show()

recognizer = sr.Recognizer()

def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        text= recognizer.recognize_google(audio_data)
        print(text)
        return text
    
transcibed_text = transcribe_audio(file_path)

# checking for accuracy
ground_truth = """My name is Ivan and I am excited to have you as part of our learning community! 
Before we get started, Iíd like to tell you a little bit about myself. Iím a sound engineer turned data scientist,
curious about machine learning and Artificial Intelligence. My professional background is primarily in media production,
with a focus on audio, IT, and communications"""

calculated_wer = wer(ground_truth, transcibed_text) # Word error rate
calculated_cer = cer(ground_truth, transcibed_text) # character error rate

print(f'Word error rate {calculated_wer}') # 34% error 
print(f'Character error rate {calculated_cer}') # 9%

'''The error rate is abit too high as we can see. In order to improve the transcription result
We can either preprocess the audio to reduce noise or use a more effective model'''

print("\n\n>>>>>>>>>>>>>> Filterring Background Noise \n\n")
# Reducing the background Noise
file_path_filtered = r'13_speech_recognition/data/filtered_speech_01.wav'
signal_filtered = librosa.effects.preemphasis(audio_signal, coef=0.97)
sf.write(file_path_filtered, signal_filtered, sample_rate)

transcibed_text_filtered = transcribe_audio(file_path_filtered)

calculated_wer = wer(ground_truth, transcibed_text_filtered) # Word error rate
calculated_cer = cer(ground_truth, transcibed_text_filtered) # character error rate

print(f'Word error rate {calculated_wer}') # 32% error 
print(f'Character error rate {calculated_cer}') # 9%

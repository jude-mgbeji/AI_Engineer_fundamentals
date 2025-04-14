from gtts import gTTS
import os

output_file = r"13_speech_recognition/data/output.mp3"

text = """My name is Jegede and I am excited to have you as part of our learning community! 
Before we get started, Iíd like to tell you a little bit about myself. Iím a sound engineer turned data scientist,
curious about machine learning and Artificial Intelligence. My professional background is primarily in media production,
with a focus on audio, IT, and communications"""

tts = gTTS(text= text, lang='en')
tts.save(output_file)

os.system(f'start {output_file}')

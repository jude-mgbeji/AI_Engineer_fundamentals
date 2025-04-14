import whisper
import csv
import os
import tempfile
import wave
from jiwer import cer, wer

file_path = r'13_speech_recognition/data/speech_01.wav'

# the model can be base, medium, large
model = whisper.load_model('base') 
result = model.transcribe(file_path)

transcribed_text = result["text"]
print(transcribed_text)

# checking for accuracy
ground_truth = """My name is Ivan and I am excited to have you as part of our learning community! 
Before we get started, Iíd like to tell you a little bit about myself. Iím a sound engineer turned data scientist,
curious about machine learning and Artificial Intelligence. My professional background is primarily in media production,
with a focus on audio, IT, and communications"""

calculated_wer = wer(ground_truth, transcribed_text) # Word error rate
calculated_cer = cer(ground_truth, transcribed_text) # character error rate

print(f'Word error rate {calculated_wer}') # 22% error 
print(f'Character error rate {calculated_cer}') # 4.6%

print("\n\n >>>>>>>>>>>>>>> Transcribing Multiple Audio Files from a Directory \n\n")

directory_path = r"13_speech_recognition/data/Recordings"

def transcribe_multiple_audio(directory_path):
    transcriptions = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(directory_path, file_name)
            result = model.transcribe(file_path)
            transcription = result["text"]
            transcriptions.append({"file_name": file_name, "transcription": transcription})
    return transcriptions
    
transcriptions = transcribe_multiple_audio(directory_path)
print(transcriptions)

print("\n\n >>>>>>>>>>>>>>> Transcribing Multiple Audio Files from a Directory \n\n")

output_file = r"13_speech_recognition/data/transcription.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Track Number', 'File Name', 'Transcription'])
    for number, transcription in enumerate(transcriptions, start=1):
        writer.writerow([number, transcription['file_name'], transcription['transcription']])
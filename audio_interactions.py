import speech_recognition as sr
import pygame
from gtts import gTTS
import spacy
import time
import os 


# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def extract_name(transcribed_text):
    if transcribed_text is None:
        return None
    # Process the transcribed text with spaCy
    doc = nlp(transcribed_text)
    # Look for entities of type 'PERSON'
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

def speak_message(message):
    # Initialize Pygame mixer
    pygame.mixer.init()
    _speak_message(message, wait_till_end=False)    

def _speak_message(message, wait_till_end=True):
    filename = (''.join(filter(str.isalpha, message)))
    filename = f"audio/{filename[:min(15,len(filename))]}.mp3"
    # if file does not yet exist, need to create it 
    if not os.path.isfile(filename):
        tts = gTTS(text=message, lang='en')
        tts.save(filename)
        # Open the file in binary mode and flush the buffer
        with open(filename, 'rb') as file:
            file.flush()
    # play the sound file
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    if not wait_till_end:
        return
    else:
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


def speak_then_record(message, duration=8):
    recognizer = sr.Recognizer() 
    _speak_message(message)
    with sr.Microphone() as source:
        print("Recording audio...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise before recording
        try:
            # Set pause_threshold to 2 seconds to stop recording after 2 seconds of silence
            recognizer.pause_threshold = 1.5
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=duration)
            print("Recording finished")
            return audio_data
        except (sr.WaitTimeoutError, sr.RequestError, sr.UnknownValueError) as e:
            print(f"An error occurred during recording: {e}")
            return None


def transcribe_audio(audio_data):
    if audio_data is None:
        return None
    # otherwise transcribe
    recognizer = sr.Recognizer()
    try:
        # Use Google Web Speech API for online speech recognition
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return ""


def greet_and_ask_name():
    # Initialize Pygame mixer
    pygame.mixer.init()
    start_time = time.time()
    recorded_audio = speak_then_record("Hello, what's your name?")
    transcribed_text = transcribe_audio(recorded_audio)
    print("Transcribed text:", transcribed_text)
    name_string = extract_name(transcribed_text)
    print("extracted name:", name_string)
    if name_string is None:
        recorded_audio = speak_then_record("please say your name again")
        transcribed_text = transcribe_audio(recorded_audio)
        print("Transcribed text:", transcribed_text)
        name_string = transcribed_text
        print("extracted name:", name_string)
        _speak_message(f"Nice to meet you {name_string}")
    else:
        _speak_message(f"Nice to meet you {name_string}")
    end_time = time.time()
    pygame.quit()
    return name_string, start_time, end_time


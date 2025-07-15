import torch
import numpy as np
import sounddevice as sd
import webrtcvad
import wave
import datetime
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#import speech_recognition as sr
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import stopwords

# Downloads (only once)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

MODEL_PATH = "best_model_CNN_and_Simple_Linear_Attention_Layer.pth"
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 10
AUDIO_UPLOAD_FOLDER = "./SavedRecordings"
os.makedirs(AUDIO_UPLOAD_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

stop_words = set(stopwords.words('english'))

def get_named_entities(pos_tags):
    named_entities = []
    chunks = ne_chunk(pos_tags)
    for chunk in chunks:
        if isinstance(chunk, Tree):
            entity = " ".join(c[0] for c in chunk)
            named_entities.append(entity)
    return named_entities

def is_valid_word(word, letter, pos_tag, named_entities):
    return (
        word
        and word[0].upper() == letter.upper()
        and word.lower() not in stop_words
        and word not in named_entities
        and word.isalpha()
        and pos_tag.startswith("NN")
    )

def score_word_task(transcription, letter='P'):
    tokens = word_tokenize(transcription)
    tagged = pos_tag(tokens)
    named_entities = get_named_entities(tagged)
    valid_words = [word for word, tag in tagged if is_valid_word(word, letter, tag, named_entities)]
    return len(valid_words), valid_words

def process_audio_and_get_prediction():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()

    # Record
    filename = f"recording_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
    filepath = os.path.join(AUDIO_UPLOAD_FOLDER, filename)
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
    sd.wait()

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording.tobytes())

    # Transcribe
    recognizer = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio_data = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio_data)
    except:
        transcription = ""

    # Spectrogram
    y, sr_ = librosa.load(filepath, sr=None)
    ms = librosa.feature.melspectrogram(y=y, sr=sr_)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.set_axis_off()
    librosa.display.specshow(log_ms, sr=sr_, ax=ax)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(img).convert('RGB')
    plt.close(fig)

    # Predict
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    label = "Dementia" if predicted.item() == 1 else "Control"

    # Verbal Fluency
    fluency_score, valid_words = score_word_task(transcription)

    return {
        "prediction": label,
        "transcription": transcription,
        "verbal_fluency_score": fluency_score,
        "valid_words": valid_words
    }

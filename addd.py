import streamlit as st  # Importing Streamlit library
import numpy as np  # For numerical operations
import sounddevice as sd  # For real-time audio recording
import wave  # For saving WAV files
import librosa  # For audio processing
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf  # For reading audio files
import io  # Needed for handling uploaded files
from tensorflow.keras.models import load_model  # For loading the trained LSTM model

# Load the LSTM model
@st.cache_resource
def load_LSTM_model():
    return load_model(r'C:\Users\mylen\Downloads\Speech-Emotion-Recognition-main (1)\Speech-Emotion-Recognition-main\LSTM_model_Date_Time_2025_02_23_06_28_01___Loss_0.09079879522323608___Accuracy_0.9652777910232544.h5')

# Extract MFCC features from audio

import numpy as np
import librosa
import io

def extract_mfcc(audio_input):
    '''Extracts MFCC features from an audio file or uploaded BytesIO object.'''

    try:
        print("🔍 Checking input type...")

        # 1️⃣ **Handle Streamlit uploaded file (BytesIO object)**
        if isinstance(audio_input, io.BytesIO):
            print("📂 Input detected as BytesIO (uploaded file). Reading...")
            audio_input.seek(0)  # Reset file pointer
            y, sr = librosa.load(audio_input, sr=None)

        # 2️⃣ **Handle recorded audio (file path)**
        elif isinstance(audio_input, str):
            print(f"📂 Input detected as file path: {audio_input}. Reading...")
            y, sr = librosa.load(audio_input, sr=None)

        else:
            raise ValueError("❌ Invalid input type. Expected file path or BytesIO.")

        # 🚨 **Check if audio is loaded correctly**
        if y is None or len(y) == 0:
            raise ValueError("❌ Audio file is empty or unreadable.")

        print(f"✅ Audio loaded! Sample rate: {sr}, Duration: {len(y) / sr:.2f} sec")

        # 3️⃣ **Extract MFCC features**
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # 🚨 **Check if MFCCs were extracted**
        if mfccs.shape[1] == 0:
            raise ValueError("❌ MFCC extraction failed. Audio might be too short or silent.")

        # 🔄 Compute mean MFCC values
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # 🚨 **Ensure 40 MFCC features are extracted**
        if len(mfccs_mean) != 40:
            raise ValueError(f"❌ Expected 40 MFCC features, but got {len(mfccs_mean)}.")

        print("✅ MFCC extraction successful!")
        return mfccs_mean

    except Exception as e:
        print(f"❌ ERROR in extract_mfcc: {e}")
        return None



# Predict emotion from audio
def predict(model, wav_filepath):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
                5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}

    test_point = extract_mfcc(wav_filepath)

    if test_point is None:
        return "Error: Could not extract MFCC features."

    test_point = np.reshape(test_point, newshape=(1, 40, 1))  # Safe reshape

    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions[0]) + 1]


# Record audio in real-time
def record_audio(filename, duration=5, samplerate=44100):
    st.write("🎙️ Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait for the recording to finish

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    st.success(f"✅ Audio recorded and saved as {filename}")

# Plot the audio waveform
def plot_waveform(audio_source):
    if isinstance(audio_source, str):  # If it's a file path
        y, sr = librosa.load(audio_source, sr=None)
    else:  # If it's an uploaded file
        y, sr = sf.read(io.BytesIO(audio_source.read()))

    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform of Audio")
    st.pyplot(fig)

# Handle audio processing (upload + recording)
# Process the recorded or uploaded audio
def process_audio(audio_source, model):
    st.audio(audio_source, format='audio/wav')

    # Ensure correct format (file path vs. uploaded file)
    plot_waveform(audio_source)  
    emotion = predict(model, audio_source)  
    st.success(f"🗣️ Detected Emotion: **{emotion}**")



# Streamlit UI
def main():
    st.image(r'C:\Users\mylen\Downloads\Speech-Emotion-Recognition-main (1)\Speech-Emotion-Recognition-main\speechrecog.jpeg')
    st.title("🎤 Speech Emotion Recognition")

    model = load_LSTM_model()  # Load LSTM model

    # Sidebar options for file upload or recording
    option = st.sidebar.radio("Choose Input Method:", ["🎙️ Record Audio", "📂 Upload Audio File"])

    if option == "🎙️ Record Audio":
        duration = st.slider("⏳ Select recording duration (seconds)", 1, 10, 5)
        if st.button("🎙️ Start Recording"):
            filename = "output.wav"
            record_audio(filename, duration)
            process_audio(filename, model)  # Use filename instead of file-like object

    elif option == "📂 Upload Audio File":
        file_uploaded = st.file_uploader("📂 Choose an audio file...", type='wav')
        if file_uploaded:
            process_audio(file_uploaded, model)

if __name__ == "__main__":
    main()

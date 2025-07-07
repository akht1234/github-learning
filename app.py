import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io
import json
from tensorflow.image import resize
from tensorflow.keras.models import load_model as keras_load_model
import streamlit.components.v1 as components

# Function to embed Lottie animation with transparent background using remote URL
def show_lottie_from_url(url, height=300):
    components.html(f"""
        <div style="display: flex; justify-content: center;">
            <lottie-player 
                src="{url}"
                background="transparent" 
                speed="1" 
                style="width: {height}px; height: {height}px;" 
                loop autoplay>
            </lottie-player>
        </div>
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    """, height=height+20)

# Load model
@st.cache_resource()
def load_model():
    return keras_load_model("Trained_model.h5")

# Preprocess audio to Mel spectrograms
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    data = []
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel = resize(np.expand_dims(mel, axis=-1), target_shape)
        data.append(mel)
    return np.array(data)

# Predict genre
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    y_avg = np.mean(y_pred, axis=0)
    top_idx = np.argmax(y_avg)
    return top_idx, y_avg[top_idx], y_avg

# Spectrogram display
def show_spectrogram(filepath):
    y, sr = librosa.load(filepath, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Mel Spectrogram')
    st.pyplot(fig)

# PDF report
def create_report(filename, genre, confidence):
    text = f"Music Genre Classification Report\n\nFile: {filename}\nGenre: {genre.capitalize()}\nConfidence: {confidence*100:.2f}%\n"
    report = io.BytesIO(text.encode('utf-8'))
    b64 = base64.b64encode(report.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="genre_report.txt">Download Report</a>'

# Page config
st.set_page_config(page_title="BeatSense AI", layout="wide")

# Sidebar
st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content { background-color: #1f2937; color: white; font-family: 'Segoe UI'; }
    </style>
""", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Navigate", ["Home", "About Project"], key="main_nav")

# Background and styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');

.stApp {
    background-image: url('https://media.istockphoto.com/id/1076840920/vector/music-background.jpg?s=612x612&w=0&k=20&c=bMG2SEUYaurIHAjtRbw7bmjLsXyT7iJUvAM5HjL3G3I=');
    background-size: cover;
    background-attachment: fixed;
    color: #f8fafc;
    font-family: 'Raleway', sans-serif;
}

h1, h2, h3, h4 {
    color: #fbbf24;
    font-weight: 700;
}

.highlight-box {
    background: #0f172a;
    padding: 2rem;
    border-radius: 15px;
    font-size: 1.4rem;
    margin: 2rem 0;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Lottie animation URL
lottie_url2 = "https://lottie.host/4231e9df-2ecd-4a66-84bd-4f7e5c1bfcc1/OEqMPX0zRK.json"
lottie_url= "https://lottie.host/7955ce9d-be37-4c67-a33f-c139ea625424/HkRGaA5fh0.json"
lottie_url3= "https://lottie.host/b3d1502c-6186-4618-b2d4-3f9e2fd40b5d/1boXYiMEWu.json"

# Home Page
if app_mode == "Home":
    st.markdown("""
        <h1 style='font-size:3rem'>🎧 BeatSense AI: Genre Identifier</h1>
        <h4 style='margin-top:-10px'>Developed by <b>Akshat Manas</b>, an ECE undergrad passionate about music tech.</h4>
    """, unsafe_allow_html=True)

    show_lottie_from_url(lottie_url, height=250)

    st.markdown("""
        <p style='font-size:1.2rem;'>Upload a music file to discover its genre using deep learning and spectrogram analysis. BeatSense AI helps musicians, researchers, and enthusiasts quickly classify and explore audio tracks intelligently.</p>
        <ul>
        <li>🎼 Powered by CNNs trained on real-world genre data</li>
        <li>📊 Outputs prediction confidence and probabilities</li>
        <li>📁 Generates a downloadable genre report</li>
        </ul>
    """, unsafe_allow_html=True)

    file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if file:
        path = f"Test_Music/{file.name}"
        with open(path, 'wb') as f:
            f.write(file.read())
        st.audio(file)

        if st.button("Predict Genre"):
            with st.spinner("Analyzing audio using spectrogram features..."):
                X = load_and_preprocess_data(path)
                genre_id, conf, all_probs = model_prediction(X)
                genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                genre = genres[genre_id]

                st.markdown(f"""
                    <div class='highlight-box'>
                        <div style='font-size:1.6rem;'>Predicted Genre:</div>
                        <div style='font-size:2.5rem; color:#facc15;'><b>{genre.capitalize()}</b></div>
                        <div style='margin-top:1rem;font-size:1.2rem;color:#38bdf8;'>Confidence: {conf*100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

                st.subheader("Mel Spectrogram Visualization")
                show_spectrogram(path)

                st.subheader("Genre Probability Distribution")
                df = pd.DataFrame({"Genre": genres, "Probability": all_probs})
                st.bar_chart(df.set_index("Genre"))

                st.markdown(create_report(file.name, genre, conf), unsafe_allow_html=True)
                show_lottie_from_url(lottie_url2, height=200)

# About
elif app_mode == "About Project":
    st.title("About BeatSense AI")
    st.markdown("""
    BeatSense AI is a powerful music genre classification tool designed to assist:

    - 🎧 Independent artists in organizing and labeling their music
    - 🧠 Researchers in analyzing audio patterns and genre traits
    - 🎼 Hobbyists curious about machine learning in music

    ### 🧠 How It Works:
    - Audio is chunked, preprocessed, and converted to Mel spectrograms (a visual representation of frequency)
    - A trained CNN model analyzes frequency patterns to predict genres
    - Probabilities for all genres are returned to visualize prediction confidence

    ### 📂 Dataset Used:
    - GTZAN Genre Collection (10 genres, 1000 songs total)
    - Each audio file: 30 seconds, sampled at 22,050 Hz

    ### 🔧 Key Features:
    - Clean UI with dark theme and Lottie animations
    - Audio upload and playback
    - Spectrogram generation with matplotlib
    - Model inference using TensorFlow
    - PDF-style genre report for download

    ### 🏁 Roadmap:
    - Add multi-label genre prediction
    - Improve UI for mobile screens
    - Integrate user feedback for better predictions
    """)
    show_lottie_from_url(lottie_url3, height=200)

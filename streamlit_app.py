import os
import tempfile

import pandas as pd
import torch
import streamlit as st
from audiorecorder import audiorecorder

from emotion_inference import EMOTION_NAMES, get_device, predict_audio

st.set_page_config(page_title="SER Emotion Tester", page_icon="🎙️", layout="wide")

st.title("Speech Emotion Recognition Tester")
st.write("Upload an audio file or record your voice, then classify the emotion with the trained Flower model.")

with st.sidebar:
    st.header("Model Status")
    st.write("Checkpoint directory: `checkpoints_flwr`")
    st.write("Device: " + get_device())
    st.caption("The app loads the latest Flower checkpoint automatically.")


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def save_recorded_audio(audio_segment) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        audio_segment.export(temp_file.name, format="wav")
        return temp_file.name


uploaded_file = st.file_uploader(
    "Upload custom audio",
    type=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
)

st.subheader("Record Your Voice")
audio_segment = audiorecorder("Start recording", "Stop recording", show_visualizer=True)

left, right = st.columns(2)
with left:
    if uploaded_file is not None:
        st.audio(uploaded_file)
with right:
    if len(audio_segment) > 0:
        st.audio(audio_segment.export().read(), format="audio/wav")

selected_path = None
source_label = None

if uploaded_file is not None:
    selected_path = save_uploaded_file(uploaded_file)
    source_label = f"uploaded file: {uploaded_file.name}"
elif len(audio_segment) > 0:
    selected_path = save_recorded_audio(audio_segment)
    source_label = "recorded voice"

if selected_path:
    st.info(f"Ready to classify {source_label}.")
    if st.button("Predict Emotion"):
        result = predict_audio(selected_path)
        st.success(f"Predicted emotion: {result['emotion'].upper()}")
        st.caption(f"Checkpoint: {result['checkpoint_path']}")
        st.caption(f"Device: {result['device']}")

        probs = pd.DataFrame(
            {
                "emotion": EMOTION_NAMES,
                "probability": result["probabilities"],
            }
        )
        st.bar_chart(probs.set_index("emotion"))

        st.write("Class probabilities")
        st.dataframe(probs, use_container_width=True)
else:
    st.warning("Upload a file or record audio to enable prediction.")

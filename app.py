import os
import tempfile

import librosa
import streamlit as st
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor


@st.cache_resource
def load_model():
    model_path = "./bird_sound_classifier_final"
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return processor, model


SELECTED_CLASSES = [
    "background",
    "Asian_openbill_stork",
    "Blue-tailed_bee-eater",
    "Common_kingfisher",
    "Eurasian_spoonbill",
    "Fulvous_whistling_duck",
    "Garganey",
    "Glossy_ibis",
    "Golden_oriole",
    "Great_egret",
    "Grey_Heron",
]


st.set_page_config(page_title="Bird Sound Classifier", page_icon="bird")
st.title("Bird Sound Classifier")
st.write("Upload a bird audio file and the model will identify the species.")

processor, model = load_model()

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "flac", "ogg"])

if audio_file is not None:
    st.audio(audio_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing audio..."):
        waveform, _ = librosa.load(tmp_path, sr=16000, mono=True)
        os.unlink(tmp_path)

        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        pred_label = model.config.id2label[pred_id]
        confidence = probs[pred_id].item() * 100

    st.success(f"Predicted: **{pred_label.replace('_', ' ')}**")
    st.metric("Confidence", f"{confidence:.2f}%")

    if confidence < 60:
        st.warning("Low confidence - result may not be reliable")

    st.subheader("Top 3 Predictions")
    sorted_probs = sorted(
        [(model.config.id2label[i], probs[i].item()) for i in range(len(probs))],
        key=lambda x: -x[1],
    )[:3]

    for name, prob in sorted_probs:
        st.progress(float(prob), text=f"{name.replace('_', ' ')}: {prob * 100:.1f}%")

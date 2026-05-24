import os
import re
import tempfile
from pathlib import Path

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


IMAGE_DIR = Path(__file__).parent / "Bird_img"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_AUDIO_SECONDS = 15

BIRD_DESCRIPTIONS = {
    "Asian_openbill_stork": "A large wetland stork with a distinctive gap between the tips of its bill. It is often seen in marshes, rice fields, and shallow wetlands, where it feeds mainly on snails and other small aquatic animals.",
    "Blue-tailed_bee-eater": "A colorful, fast-flying bird with green body feathers, a blue tail, and a dark eye stripe. It catches bees, dragonflies, and other flying insects in open country, farmland, and areas near water.",
    "Common_kingfisher": "A small but striking bird with bright blue upperparts and orange underparts. It usually sits quietly near ponds, rivers, or streams before diving into the water to catch small fish.",
    "Eurasian_spoonbill": "A white wading bird recognized by its long, spoon-shaped bill. It walks through shallow water sweeping its bill from side to side to find fish, insects, crustaceans, and other small prey.",
    "Fulvous_whistling_duck": "A warm brown duck with long legs and an upright posture compared with many other ducks. It often forms noisy flocks around lakes, wetlands, and flooded fields, and its call has a clear whistling sound.",
    "Garganey": "A small migratory duck that visits wetlands, ponds, and flooded grasslands. The male is especially noticeable in breeding season because of the bold pale stripe above the eye, while females are more softly patterned brown.",
    "Glossy_ibis": "A dark wading bird with a long curved bill and shiny bronze-green tones in good light. It feeds in wetlands, muddy shallows, and flooded fields, probing the ground for insects, worms, and small aquatic animals.",
    "Golden_oriole": "A bright yellow songbird that usually stays high in leafy trees, making it easier to hear than to see. Its rich, fluting call is distinctive, and it feeds on insects, fruit, and berries.",
    "Great_egret": "A tall, elegant white heron with a long neck, black legs, and a yellow bill outside the breeding season. It hunts patiently in shallow water, slowly stalking fish, frogs, and other small prey.",
    "Grey_Heron": "A large grey wading bird with a long neck, dagger-like bill, and slow, heavy wingbeats. It is commonly found beside rivers, ponds, lakes, and wetlands, often standing still while waiting to strike at fish.",
}


def normalize_bird_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


@st.cache_data
def load_bird_images():
    if not IMAGE_DIR.exists():
        return {}

    images = {}
    for image_path in IMAGE_DIR.iterdir():
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images[normalize_bird_name(image_path.stem)] = image_path
    return images


def display_bird_name(label):
    return label.replace("_", " ")


def prepare_waveform(audio_path):
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    waveform, _ = librosa.effects.trim(waveform, top_db=25)

    max_samples = 16000 * MAX_AUDIO_SECONDS
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    return waveform


@st.cache_data(show_spinner=False)
def predict_audio(audio_bytes, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        waveform = prepare_waveform(tmp_path)
    finally:
        os.unlink(tmp_path)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()
    pred_label = model.config.id2label[pred_id]
    confidence = probs[pred_id].item() * 100
    top_predictions = sorted(
        [(model.config.id2label[i], probs[i].item()) for i in range(len(probs))],
        key=lambda x: -x[1],
    )[:3]

    return pred_label, confidence, top_predictions


st.set_page_config(page_title="Bird Sound Classifier", page_icon="bird")
st.title("Bird Sound Classifier")
st.write("Upload a bird audio file and the model will identify the species.")
st.warning(
    "Disclaimer: This model may not always predict correctly. Results can be affected by poor sound quality, "
    "background noise, overlapping bird calls, very short clips, or species that sound similar."
)

processor, model = load_model()
bird_images = load_bird_images()

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "flac", "ogg"])

if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    suffix = os.path.splitext(audio_file.name)[1]

    st.audio(audio_bytes)
    with st.spinner("Analyzing audio..."):
        pred_label, confidence, sorted_probs = predict_audio(audio_bytes, suffix)

    st.success(f"Predicted: **{display_bird_name(pred_label)}**")
    st.metric("Confidence", f"{confidence:.2f}%")

    bird_image = bird_images.get(normalize_bird_name(pred_label))
    if bird_image:
        st.image(str(bird_image), caption=display_bird_name(pred_label), width=340)
    elif pred_label.lower() != "background":
        st.info("No matching bird image found in the Bird_img folder.")

    description = BIRD_DESCRIPTIONS.get(pred_label)
    if description:
        st.write(description)

    if confidence < 60:
        st.warning("Low confidence - result may not be reliable")

    st.subheader("Top 3 Predictions")

    for name, prob in sorted_probs:
        st.progress(float(prob), text=f"{display_bird_name(name)}: {prob * 100:.1f}%")

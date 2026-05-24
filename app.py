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
st.markdown(
    """
    <style>
        .stApp {
            background: #f6f8f4;
        }

        .block-container {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .app-header {
            border-left: 6px solid #3b7f5f;
            padding: 0.15rem 0 0.3rem 1rem;
            margin-bottom: 1.25rem;
        }

        .app-header h1 {
            color: #18362c;
            font-size: 2.2rem;
            line-height: 1.15;
            margin: 0;
        }

        .app-header p {
            color: #4c5f57;
            font-size: 1rem;
            margin: 0.45rem 0 0;
        }

        .section-panel {
            background: #ffffff;
            border: 1px solid #dfe7df;
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(35, 59, 48, 0.06);
        }

        .section-panel h3 {
            color: #18362c;
            font-size: 1.05rem;
            margin: 0 0 0.65rem;
        }

        .disclaimer {
            background: #fff8e6;
            border: 1px solid #f0d48b;
            border-left: 5px solid #c58b17;
            border-radius: 8px;
            color: #5e4514;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
        }

        .prediction-title {
            color: #18362c;
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0 0 0.2rem;
        }

        .prediction-label {
            color: #3b7f5f;
        }

        .bird-description {
            color: #40534b;
            font-size: 1rem;
            line-height: 1.55;
            margin-top: 0.75rem;
        }

        .stProgress > div > div > div > div {
            background-color: #3b7f5f;
        }

        div[data-testid="stMetric"] {
            background: #eef5ef;
            border: 1px solid #d5e4d6;
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }

        div[data-testid="stFileUploader"] {
            background: #fbfcfb;
            border: 1px dashed #a8b9ad;
            border-radius: 8px;
            padding: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>Bird Sound Classifier</h1>
        <p>Upload a bird audio clip to identify the most likely species.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="disclaimer">
        <strong>Model disclaimer:</strong> Predictions may be inaccurate when audio quality is poor,
        background noise is high, clips are very short, calls overlap, or species sound similar.
    </div>
    """,
    unsafe_allow_html=True,
)

processor, model = load_model()
bird_images = load_bird_images()

st.markdown('<div class="section-panel"><h3>Audio Upload</h3>', unsafe_allow_html=True)
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])
st.markdown("</div>", unsafe_allow_html=True)

if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    suffix = os.path.splitext(audio_file.name)[1]

    st.audio(audio_bytes)
    with st.spinner("Analyzing audio..."):
        pred_label, confidence, sorted_probs = predict_audio(audio_bytes, suffix)

    bird_image = bird_images.get(normalize_bird_name(pred_label))
    description = BIRD_DESCRIPTIONS.get(pred_label)

    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    image_col, result_col = st.columns([0.9, 1.4], vertical_alignment="top")

    with image_col:
        if bird_image:
            st.image(str(bird_image), caption=display_bird_name(pred_label), width=300)
        elif pred_label.lower() != "background":
            st.info("No matching bird image found in the Bird_img folder.")

    with result_col:
        st.markdown(
            f"""
            <p class="prediction-title">
                Predicted: <span class="prediction-label">{display_bird_name(pred_label)}</span>
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence:.2f}%")
        if description:
            st.markdown(f'<p class="bird-description">{description}</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if confidence < 60:
        st.warning("Low confidence - result may not be reliable")

    st.markdown('<div class="section-panel"><h3>Top 3 Predictions</h3>', unsafe_allow_html=True)

    for name, prob in sorted_probs:
        st.progress(float(prob), text=f"{display_bird_name(name)}: {prob * 100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

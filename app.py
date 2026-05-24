import os
import re
import tempfile
from pathlib import Path

import librosa
import streamlit as st
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor


st.set_page_config(page_title="Bird Sound Classifier", page_icon="bird", layout="wide")


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


def confidence_status(confidence):
    if confidence >= 80:
        return "Strong match"
    if confidence >= 60:
        return "Moderate match"
    return "Needs review"


def render_prediction_bar(label, probability):
    percent = probability * 100
    st.markdown(
        f"""
        <div class="prediction-row">
            <div class="prediction-row-top">
                <span>{display_bird_name(label)}</span>
                <strong>{percent:.1f}%</strong>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: {percent:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


st.markdown(
    """
    <style>
        .stApp {
            background: #f4f7f3;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 3.25rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, p {
            letter-spacing: 0;
        }

        .hero {
            background: #173c32;
            border-radius: 8px;
            color: #ffffff;
            padding: 1.45rem 1.6rem;
            margin-bottom: 1rem;
            border: 1px solid #0f2d26;
        }

        .eyebrow {
            color: #b8d7c8;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }

        .hero h1 {
            color: #ffffff;
            font-size: 2.35rem;
            line-height: 1.12;
            margin: 0;
        }

        .hero p {
            color: #dcebe4;
            font-size: 1rem;
            margin: 0.55rem 0 0;
            max-width: 760px;
        }

        .disclaimer {
            background: #fff8e6;
            border: 1px solid #f0d48b;
            border-left: 6px solid #c58b17;
            border-radius: 8px;
            color: #5e4514;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
        }

        .prediction-title {
            color: #18362c;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.15;
            margin: 0 0 0.6rem;
        }

        .prediction-label {
            color: #3b7f5f;
        }

        .status-pill {
            display: inline-block;
            background: #e8f3ed;
            border: 1px solid #cde0d4;
            border-radius: 999px;
            color: #245844;
            font-size: 0.82rem;
            font-weight: 700;
            padding: 0.28rem 0.7rem;
            margin-bottom: 0.8rem;
        }

        .bird-description {
            color: #40534b;
            font-size: 1rem;
            line-height: 1.55;
            margin-top: 0.75rem;
        }

        div[data-testid="stMetric"] {
            background: #f3f8f4;
            border: 1px solid #dce8dd;
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }

        div[data-testid="stFileUploader"] {
            padding-top: 0.25rem;
        }

        .prediction-row {
            margin: 0.75rem 0;
        }

        .prediction-row-top {
            align-items: center;
            color: #2f4039;
            display: flex;
            font-size: 0.95rem;
            justify-content: space-between;
            margin-bottom: 0.3rem;
        }

        .bar-track {
            background: #e7eee8;
            border-radius: 999px;
            height: 0.68rem;
            overflow: hidden;
            width: 100%;
        }

        .bar-fill {
            background: #3b7f5f;
            border-radius: 999px;
            height: 100%;
        }

        .empty-state {
            background: #ffffff;
            border: 1px solid #dce5dd;
            border-radius: 8px;
            color: #52645d;
            padding: 1rem;
        }

        .small-note {
            color: #60716a;
            font-size: 0.9rem;
            margin-top: 0.7rem;
        }

        section[data-testid="stFileUploaderDropzone"] {
            background: #fbfdfb;
            border-color: #a8b9ad;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            box-shadow: 0 10px 22px rgba(24, 54, 44, 0.06);
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.85rem;
        }

        h3 a {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Audio recognition</div>
        <h1>Bird Sound Classifier</h1>
        <p>Identify wetland and woodland bird species from short sound recordings, with image previews, confidence scores, and quick species notes.</p>
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

left_col, right_col = st.columns([0.95, 1.05], gap="medium")

with left_col:
    with st.container(border=True):
        st.subheader("Upload Audio")
        audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])
        st.caption("Supported formats: WAV, MP3, FLAC, and OGG. Shorter clips with clear calls usually work best.")

with right_col:
    with st.container(border=True):
        st.subheader("Model Setup")
        st.write(
            f"Recognizes {len(SELECTED_CLASSES) - 1} bird species plus background audio. "
            f"Long clips are trimmed to the first {MAX_AUDIO_SECONDS} seconds for faster analysis."
        )

if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    suffix = os.path.splitext(audio_file.name)[1]

    with st.spinner("Analyzing audio..."):
        pred_label, confidence, sorted_probs = predict_audio(audio_bytes, suffix)

    bird_image = bird_images.get(normalize_bird_name(pred_label))
    description = BIRD_DESCRIPTIONS.get(pred_label, "No description is available for this prediction.")

    with left_col:
        with st.container(border=True):
            st.subheader("Audio Preview")
            st.audio(audio_bytes)

        with st.container(border=True):
            st.subheader("Species Preview")
            if bird_image:
                st.image(str(bird_image), caption=display_bird_name(pred_label), width=280)
            elif pred_label.lower() != "background":
                st.info("No matching bird image found in the Bird_img folder.")
            st.markdown(f'<p class="bird-description">{description}</p>', unsafe_allow_html=True)

    with right_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="status-pill">{confidence_status(confidence)}</div>
                <div class="prediction-title">
                    <span class="prediction-label">{display_bird_name(pred_label)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric_col, time_col = st.columns(2)
            with metric_col:
                st.metric("Confidence", f"{confidence:.2f}%")
            with time_col:
                st.metric("Analysis Window", f"{MAX_AUDIO_SECONDS}s max")

        with st.container(border=True):
            st.subheader("Top Matches")
            for name, prob in sorted_probs:
                render_prediction_bar(name, prob)

    if confidence < 60:
        st.warning("Low confidence - result may not be reliable")
else:
    with left_col:
        st.markdown(
            """
            <div class="empty-state">
                Upload an audio clip to see the prediction, confidence score, species image, and the top matching bird calls.
            </div>
            """,
            unsafe_allow_html=True,
        )

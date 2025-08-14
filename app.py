import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import requests
from streamlit_lottie import st_lottie
import random

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="💬", layout="wide")

# ----------------------------
# Function to load Lottie animations from URL
# ----------------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ----------------------------
# Lottie animation URLs for each emotion
# ----------------------------
lottie_urls = {
    "joy": "https://lottie.host/5ddc3d2b-734f-43f2-ace4-e06d74a705d4/mRP6BlTKoD.json",
    "anger": "https://lottie.host/c127e6a2-381d-4a6f-bc5b-fb9c7be80341/wjHRqKwvxA.json",
    "sadness": "https://lottie.host/2e037edb-6bba-471a-91f3-3e1fd2a2a56b/FaHtZtrHTu.json",
    "surprise": "https://lottie.host/8012f70a-e6e3-4235-8c6f-44c4c85b87a8/AtVlBu0DQQ.json",
    "fear": "https://lottie.host/f2ed9333-cc9d-444a-962f-c5eb3d3b3a6b/DtJAVq3ObW.json",
    "disgust": "https://lottie.host/fddda390-e46a-4a36-b254-40a7c50dd168/yPOXijFxOd.json",
    "neutral": "https://lottie.host/49f7921e-3b88-4638-8722-58c558bda00f/xmn9VPiJxz.json",
}

# ----------------------------
# Hugging Face Authentication (for private models)
# ----------------------------


# ----------------------------
# Load emotion classification pipeline
# ----------------------------
@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

classifier = load_classifier()

# ----------------------------
# UI Styling (Base)
# ----------------------------
st.markdown("""
    <style>
        .stTextArea textarea {
            border-radius: 12px;
            padding: 10px;
            border: 2px solid #c2d1f0;
            font-size: 16px;
        }
        .stButton button {
            background-color: #4a7cff;
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
        }
        .stButton button:hover {
            background-color: #345dcc;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper functions
# ----------------------------
def set_bg_color(color):
    """Smooth background color transition"""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
            transition: background-color 1s ease;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def show_emoji(emoji, animation_speed="2s", extra_css=""):
    """Random corner bouncing emoji"""
    start_corner = random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
    corner_pos = {
        "top-left": "top: 0; left: 0;",
        "top-right": "top: 0; right: 0;",
        "bottom-left": "bottom: 0; left: 0;",
        "bottom-right": "bottom: 0; right: 0;",
    }
    st.markdown(
        f"""
        <style>
        @keyframes bounce {{
            0%   {{ transform: translate(0, 0); }}
            50%  {{ transform: translate(50px, -50px); }}
            100% {{ transform: translate(0, 0); }}
        }}
        .emoji {{
            position: fixed;
            font-size: 5rem;
            {corner_pos[start_corner]}
            animation: bounce {animation_speed} infinite;
            z-index: 999;
            {extra_css}
        }}
        </style>
        <div class="emoji">{emoji}</div>
        """,
        unsafe_allow_html=True
    )

def load_lottie_url(url):
    """Load a Lottie animation from URL"""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URLs
lottie_tears = "https://lottie.host/6e1e2ff2-b6f8-4a07-8c92-f0cb6af609d6/tears.json"
lottie_lightning = "https://lottie.host/79d56c8b-2a54-4ffb-8c42-ef534c3ddda7/lightning.json"

# ----------------------------
# Title & Input
# ----------------------------
st.title("🎭 Emotion Detector")
st.write("Describe how you're feeling, and let AI reveal the emotion behind your words.")
user_input = st.text_area("Type your thoughts here:", height=150)

# ----------------------------
# Analyze Emotion
# ----------------------------
if st.button("🎯 Detect Emotion"):
    if user_input.strip() != "":
        with st.spinner("Analyzing your emotion..."):
            result = classifier(user_input)[0]
            label = result['label'].lower()
            score = result['score']

            # Show Result
            st.markdown(f"### 🧠 Emotion: `{label.title()}`")
            st.markdown(f"### 🔍 Confidence: `{score:.2f}`")

            # Effects based on emotion
            if label == "joy":
                set_bg_color("#FFFACD")  # light yellow
                show_emoji("😊", "1.5s", "text-shadow: 0 0 20px yellow;")

            elif label == "sadness":
                set_bg_color("#D3D3D3")  # grey
                show_emoji("😢", "3s", "filter: drop-shadow(2px 4px 6px blue);")
                tears_json = load_lottie_url(lottie_tears)
                if tears_json:
                    st_lottie(tears_json, height=200, key="sad_tears")

            elif label == "anger":
                set_bg_color("#FFD1D1")  # reddish
                show_emoji("😡", "0.8s", "animation-timing-function: steps(4, end);")
                lightning_json = load_lottie_url(lottie_lightning)
                if lightning_json:
                    st_lottie(lightning_json, height=250, key="angry_lightning")


            else:
                set_bg_color("#f6f9fc")  # default
                st.empty()

    else:
        st.warning("🚨 Please enter some text to analyze.")

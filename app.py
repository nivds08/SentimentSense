import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import requests
from streamlit_lottie import st_lottie

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
# If your model is private, uncomment this:
# login(token=st.secrets["HF_TOKEN"])

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
# UI Styling
# ----------------------------
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f6f9fc;
        }
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

            # Show corresponding Lottie animation
            if label in lottie_urls:
                lottie_json = load_lottie_url(lottie_urls[label])
                if lottie_json:
                    st_lottie(lottie_json, height=300, key=label)
    else:
        st.warning("🚨 Please enter some text to analyze.")

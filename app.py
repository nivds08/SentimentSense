import streamlit as st
from transformers import pipeline

# Load emotion classification pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

# Streamlit UI
st.set_page_config(page_title="Emotion Detector")
st.title("💬 Text Emotion Detection")
st.write("Enter a sentence below to detect its emotion:")

# Input text
user_input = st.text_area("Your Text", "")

# Analyze button
if st.button("Detect Emotion"):
    if user_input.strip() != "":
        with st.spinner("Analyzing..."):
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']
            st.markdown(f"**Emotion:** `{label}`")
            st.markdown(f"**Confidence:** `{score:.2f}`")
    else:
        st.warning("Please enter some text first.")

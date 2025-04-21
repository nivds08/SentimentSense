import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")


# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer")
st.title("💬 Text Sentiment Analysis")
st.write("Enter a sentence below to see its sentiment:")

# Input text
user_input = st.text_area("Your Text", "")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        with st.spinner("Analyzing..."):
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']
            st.markdown(f"**Sentiment:** `{label}`")
            st.markdown(f"**Confidence:** `{score:.2f}`")
    else:
        st.warning("Please enter some text first.")
        print("App loaded successfully.")


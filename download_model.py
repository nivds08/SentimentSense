from transformers import pipeline

# This will download and cache the sentiment model
print("Downloading model...")
classifier = pipeline("sentiment-analysis")
print("✅ Model downloaded successfully!")

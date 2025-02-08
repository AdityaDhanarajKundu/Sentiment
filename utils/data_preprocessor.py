import re


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

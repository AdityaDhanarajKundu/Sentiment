from transformers import pipeline


# Function to load the sentiment model
def load_sentiment_model():
    model_path = "sentiment_model"
    sentiment_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path,
                                  top_k=None)
    return sentiment_pipeline

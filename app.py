from flask import Flask, request, jsonify
from utils.model_loader import load_sentiment_model
from utils.data_preprocessor import preprocess_text

# Initialize Flask app
app = Flask(__name__)

# Load the sentiment model
sentiment_pipeline = load_sentiment_model()
print("Model loaded successfully!")

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Mental Health Companion AI API!"})


# Endpoint for sentiment analysis
@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    try:
        # Get the text input from the request
        data = request.json
        if "text" not in data or not data["text"]:
            return jsonify({"error": "Invalid input. 'text' field is required."}), 400

        text = data["text"]

        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Get sentiment predictions
        result = sentiment_pipeline(preprocessed_text)

        formatted_predictions = [
            {
                "Emotion": emotion_labels[int(pred["label"].split("_")[1])],
                "Score": pred["score"]
            }
            for pred in sorted(result[0], key= lambda x: x["score"], reverse=True)
        ]

        # Format the response
        response = {
            "input": text,
            "preprocessed_input": preprocessed_text,
            "predictions": formatted_predictions
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

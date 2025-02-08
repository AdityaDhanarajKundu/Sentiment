from transformers import pipeline

# Load the fine-tuned model
model_path = "./sentiment_model"
sentiment_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path,
                              top_k=None)

# Test the model with example
if __name__ == "__main__":
    test_input = "I am feeling heartbroken"
    result = sentiment_pipeline(test_input)
    print("Input: ", test_input)
    print("Result: ", result)

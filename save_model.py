from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
import numpy as np
from sklearn.metrics import classification_report

# Load the instance of goEmotions dataset, it has 27 emotion labels and 1 neutral label
dataset = load_dataset("go_emotions", "simplified")

# Defining the emotion labels
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Load Tokenizer and Model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=len(emotion_labels))
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Convert multi-label targets to single-label (selecting the first label for simplicity)
def process_labels(example):
    example["labels"] = example["labels"][0] if isinstance(example["labels"], list) else example[
        "labels"]
    return example


# Apply the processing to the dataset
dataset = dataset.map(process_labels)


# Tokenize function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text", "id"])  # Remove unnecessary columns
tokenized_datasets = tokenized_datasets.rename_column("labels", "label")  # Align column name
tokenized_datasets.set_format("torch")

# Split Dataset
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Print the first example from the train dataset to verify the labels
print(train_dataset[0])


# Compute Metrics
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    return classification_report(labels, predictions, target_names=emotion_labels, output_dict=True)


# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

print("Fine-tuned model and tokenizer saved to './sentiment_model'")

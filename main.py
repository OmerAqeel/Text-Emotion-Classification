from src.data_preprocessing import load_data, preprocess_texts, encode_labels
from src.model_training import build_model, train_model
from src.evaluation import evaluate_model
from src.predict import predict_emotion
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = load_data('datasets/train.txt')
texts = data["text"].tolist()
labels = data["Emotions"].tolist()

tokenizer, padded_sequences, max_length = preprocess_texts(texts)
encoder, one_hot_labels = encode_labels(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.2, random_state=42)

# Build and train model
model = build_model(len(tokenizer.word_index) + 1, max_length)
history = train_model(model, X_train, y_train, X_test, y_test)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Classify user input
user_text = input("Enter a text to classify its emotion: ")
predicted_emotion = predict_emotion(model, tokenizer, encoder, user_text, max_length)
print(f"The predicted emotion for the input text is: {predicted_emotion}")



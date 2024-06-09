from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_input_text(tokenizer, text, max_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence

def predict_emotion(model, tokenizer, encoder, text, max_length):
    padded_sequence = preprocess_input_text(tokenizer, text, max_length)
    prediction = model.predict(padded_sequence)
    predicted_label = encoder.inverse_transform([prediction.argmax(axis=-1)[0]])
    return predicted_label[0]

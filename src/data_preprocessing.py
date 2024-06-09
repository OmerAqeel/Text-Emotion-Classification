import pandas as pd
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path, sep=";")
    data.columns = ["text", "Emotions"]
    return data

def preprocess_texts(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = max([len(s.split()) for s in texts])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return tokenizer, padded_sequences, max_length

def encode_labels(labels):
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    one_hot_labels = tf.keras.utils.to_categorical(labels)
    return encoder, one_hot_labels
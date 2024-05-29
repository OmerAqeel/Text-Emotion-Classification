import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer  # to tokenize the text data into a format that can be used by the neural network
from tensorflow.keras.preprocessing.sequence import pad_sequences  # to pad the sequences so they are all the same size
from sklearn.preprocessing import LabelEncoder  # to convert the labels into a format that can be used by the neural network
from sklearn.model_selection import train_test_split  # to split the data into training and testing data
from tensorflow.keras.models import Sequential  # this is the type of neural network we will be using
from tensorflow.keras.layers import Embedding, Flatten, Dense  # these are the layers we will be using in our neural network

# Read the data
data = pd.read_csv('datasets/train.txt', sep=";")
data.columns = ["text", "Emotions"]  # set the column names
print(data.head())  # print the first few rows of the data

texts = data["text"].tolist()  # get the text data
labels = data["Emotions"].tolist()  # get the label data

# Tokenize the data
tokenizer = Tokenizer() # create the tokenizer
tokenizer.fit_on_texts(texts) # fit the tokenizer on the text

sequences = tokenizer.texts_to_sequences(texts) # convert the text to sequences
max_length = max([len(s.split()) for s in texts]) # get the max length of the sequences (the max number of words in a text) that will be used for padding
padded_sequences = pad_sequences(sequences, maxlen=max_length) # pad the sequences

# Encode the string labels into integers
encoder = LabelEncoder() # create the encoder
labels = encoder.fit_transform(labels) # fit the encoder on the labels

one_hot_labels = tf.keras.utils.to_categorical(labels) # convert the labels to one hot encoding

# Split the data into training and testing data
# padded sequence is the input data
# one_hot_labels is the array of labels corresponding to the input data, which will also be split into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.2, random_state=42) # split the data

# Define the model
model = Sequential() # create the model
model.add(Embedding(len(tokenizer.word_index)+1, 32, input_length=max_length)) # add the embedding layer

model.add(Flatten()) # add the flatten layer
model.add(Dense(units=128, activation='relu')) # add the dense layer
model.add(Dense(units=len(one_hot_labels[0]), activation='softmax')) # add the output layer

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # compile the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)) # fit the model



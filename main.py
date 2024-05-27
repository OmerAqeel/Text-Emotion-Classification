import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import tensorflow as tf  # this is the deep learning library we are using
from tensorflow.keras.preprocessing.text import Tokenizer  # this module is used to tokenize the text data into a format that can be used by the neural network
from tensorflow.keras.preprocessing.sequence import pad_sequences  # this module is used to pad the sequences so they are all the same size
from sklearn.preprocessing import LabelEncoder  # this module is used to convert the labels into a format that can be used by the neural network
from sklearn.model_selection import train_test_split  # this module is used to split the data into training and testing data
from tensorflow.keras.models import Sequential  # this is the type of neural network we will be using
from tensorflow.keras.layers import Embedding, Flatten, Dense  # these are the layers we will be using in our neural network

# Read the data
data = pd.read_csv('datasets/train.txt', sep=";")
data.columns = ["text", "Emotions"]  # set the column names
print(data.head())  # print the first few rows of the data

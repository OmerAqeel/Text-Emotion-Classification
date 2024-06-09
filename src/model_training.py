from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

def build_model(vocab_size, input_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=input_length))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(6, activation='softmax'))  
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history
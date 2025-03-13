import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from config import SEQUENCE_LENGTH, FEATURES

def build_model():
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQUENCE_LENGTH, len(FEATURES)), return_sequences=False),
        RepeatVector(SEQUENCE_LENGTH),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(len(FEATURES)))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()

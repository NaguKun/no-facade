from data_processing import load_data, preprocess_data
from model import build_model
from config import EPOCHS, BATCH_SIZE, VALIDATION_SPLIT

def train_model():
    df = load_data()
    sequences, scaler = preprocess_data(df)
    
    model = build_model()
    model.fit(sequences, sequences, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

    model.save("lstm_autoencoder.h5")
    print("Model saved!")

if __name__ == "__main__":
    train_model()

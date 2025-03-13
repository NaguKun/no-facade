import numpy as np
import tensorflow as tf
from data_processing import load_data, preprocess_data
from config import SEQUENCE_LENGTH, FEATURES, THRESHOLD_PERCENTILE

def detect_anomalies():
    df = load_data()
    sequences, scaler = preprocess_data(df)

    model = tf.keras.models.load_model("lstm_autoencoder.h5")
    predicted = model.predict(sequences)
    
    mse = np.mean(np.abs(predicted - sequences), axis=(1, 2))
    threshold = np.percentile(mse, THRESHOLD_PERCENTILE)
    
    df["anomaly_score"] = np.append(mse, [0] * SEQUENCE_LENGTH)
    df["anomaly"] = df["anomaly_score"] > threshold

    print(f"Anomalies detected: {df['anomaly'].sum()}")
    return df[df["anomaly"]]

if __name__ == "__main__":
    anomalies = detect_anomalies()
    print(anomalies[["date_time", "trans_no", "credit", "debit", "detail"]])

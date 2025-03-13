import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import SEQUENCE_LENGTH, FEATURES

def load_data(filename="chuyen_khoan.csv"):
    df = pd.read_csv(filename, parse_dates=["date_time"])
    df.sort_values("date_time", inplace=True)
    return df

def preprocess_data(df):
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    
    sequences = []
    for i in range(len(df) - SEQUENCE_LENGTH):
        sequences.append(df[FEATURES].iloc[i : i + SEQUENCE_LENGTH].values)

    return np.array(sequences), scaler  # Return the scaler to inverse transform the data later

if __name__ == "__main__":
    df = load_data()
    sequences, _ = preprocess_data(df)
    print("Processed data shape:", sequences.shape)

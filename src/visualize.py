import matplotlib.pyplot as plt
from detect import detect_anomalies
from data_processing import load_data

def plot_anomalies():
    df = detect_anomalies()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["date_time"], df["anomaly_score"], label="Anomaly Score")
    plt.axhline(y=df["anomaly_score"].max(), color="r", linestyle="--", label="Threshold")
    plt.scatter(df[df["anomaly"]]["date_time"], df[df["anomaly"]]["anomaly_score"], color="red", label="Anomalies")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_anomalies()

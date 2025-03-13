from train import train_model
from detect import detect_anomalies
from visualize import plot_anomalies

if __name__ == "__main__":
    print("Training model...")
    train_model()
    
    print("Detecting anomalies...")
    anomalies = detect_anomalies()
    print(anomalies[["date_time", "trans_no", "credit", "debit", "detail"]])

    print("Visualizing results...")
    plot_anomalies()

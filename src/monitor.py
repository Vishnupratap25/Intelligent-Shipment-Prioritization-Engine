import pandas as pd
import numpy as np
import os

def monitor_predictions(prediction_file):

    df = pd.read_csv(prediction_file)

    print("\n📊 Monitoring Report")
    print("-" * 40)

    print("\nRisk Distribution:")
    print(df["Risk_Level"].value_counts())

    print("\nAverage Failure Probability:")
    print(round(df["Failure_Probability"].mean(), 4))

    print("\nHigh Risk Percentage:")
    high_pct = (df["Risk_Level"] == "High").mean() * 100
    print(f"{round(high_pct, 2)}%")

    print("\nPrediction Balance:")
    print(df["Prediction_Label"].value_counts())


if __name__ == "__main__":
    monitor_predictions("outputs/prediction_results.csv")
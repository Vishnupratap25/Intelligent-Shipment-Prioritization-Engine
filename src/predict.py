import joblib
import pandas as pd
import numpy as np
import os
import sys

# -------------------------------------------------
# Fix Path So Both CLI and Streamlit Work
# -------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Now imports work safely
from src.data_preprocessing import drop_irrelevant_columns
from src.feature_engineering import create_time_features


# -------------------------------------------------
# Absolute Paths
# -------------------------------------------------

MODEL_PATH = os.path.join(PROJECT_ROOT, "models/sla_binary_stable_model_20260224_1642.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models/label_encoders.pkl")
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "models/failure_threshold.pkl")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# -------------------------------------------------
# Preprocessing
# -------------------------------------------------

def preprocess_input(df):

    df = drop_irrelevant_columns(df)

    if "IST_svc_commit_tmstp" in df.columns:
        df["IST_svc_commit_tmstp"] = pd.to_datetime(
            df["IST_svc_commit_tmstp"], errors="coerce"
        )

    df = create_time_features(df)

    df = df.fillna("Unknown")

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            mask = df[col].isin(le.classes_)
            df.loc[mask, col] = le.transform(df.loc[mask, col])
            df.loc[~mask, col] = 0

    drop_cols = [
        "commit_status",
        "target",
        "target_binary",
        "IST_svc_commit_tmstp",
        "time_diff_hours",
        "last_scan"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


# -------------------------------------------------
# Prediction
# -------------------------------------------------

def predict_failure(input_path):

    print("📥 Loading Input Data...")
    df = pd.read_csv(input_path, sep="\t", low_memory=False)

    print("⚙ Preprocessing...")
    processed_df = preprocess_input(df)

    print("🤖 Generating Predictions...")
    probabilities = model.predict_proba(processed_df)[:, 1]
    predictions = (probabilities > threshold).astype(int)

    result_df = pd.DataFrame({
        "Failure_Probability": probabilities,
        "Prediction": predictions
    })

    result_df["Prediction_Label"] = result_df["Prediction"].map({
        0: "Ontime",
        1: "Failure"
    })

    result_df["Risk_Level"] = pd.cut(
        probabilities,
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    return result_df


# -------------------------------------------------
# CLI Execution
# -------------------------------------------------

if __name__ == "__main__":

    INPUT_PATH = os.path.join(PROJECT_ROOT, "data/raw/data.tsv")

    results = predict_failure(INPUT_PATH)

    print("\n📊 Risk Summary:")
    print(results["Risk_Level"].value_counts())

    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    results.to_csv(
        os.path.join(PROJECT_ROOT, "outputs/prediction_results.csv"),
        index=False
    )

    print("\n✅ Predictions saved successfully.")
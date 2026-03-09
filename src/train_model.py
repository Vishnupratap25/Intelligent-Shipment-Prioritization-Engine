import os
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from data_preprocessing import load_data, clean_target, drop_irrelevant_columns
from feature_engineering import create_time_features


# -------------------------------------------------
# Sliding Time Split (Recent-Focused)
# -------------------------------------------------

def sliding_time_split(df):

    df = df.sort_values("IST_svc_commit_tmstp")

    n = len(df)

    train_start = int(n * 0.20)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    train_df = df.iloc[train_start:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# -------------------------------------------------
# Threshold Optimization
# -------------------------------------------------

def optimize_threshold(y_true, probs):

    best_thresh = 0.5
    best_f1 = 0

    for thresh in np.arange(0.30, 0.71, 0.02):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    print("🚀 Loading Data...")
    df = load_data("data/raw/data.tsv")

    print("✅ Cleaning Target...")
    df = clean_target(df)

    print("\n🎯 Original Target Distribution:")
    print(df["target"].value_counts())

    # -------------------------------------------------
    # Binary Target
    # -------------------------------------------------

    df["target_binary"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

    print("\n🎯 Binary Target Distribution:")
    print(df["target_binary"].value_counts())

    print("✅ Dropping Irrelevant Columns...")
    df = drop_irrelevant_columns(df)

    print("✅ Converting Timestamp...")
    df["IST_svc_commit_tmstp"] = pd.to_datetime(
        df["IST_svc_commit_tmstp"], errors="coerce"
    )

    print("✅ Creating Time Features...")
    df = create_time_features(df)

    print("✅ Handling Missing Values...")
    df = df.fillna("Unknown")

    print("\n📊 Dataset Shape:", df.shape)

    # -------------------------------------------------
    # Sliding Split
    # -------------------------------------------------

    train_df, val_df, test_df = sliding_time_split(df)

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    # -------------------------------------------------
    # Encode Categoricals
    # -------------------------------------------------

    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

    if "commit_status" in categorical_cols:
        categorical_cols.remove("commit_status")

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))

        val_df[col] = val_df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

        test_df[col] = test_df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

        label_encoders[col] = le

    drop_cols = [
        "commit_status",
        "target",
        "target_binary",
        "IST_svc_commit_tmstp",
        "time_diff_hours",
        "last_scan"
    ]

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df["target_binary"]

    X_val = val_df.drop(columns=drop_cols, errors="ignore")
    y_val = val_df["target_binary"]

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df["target_binary"]

    # -------------------------------------------------
    # Class Weighting (Very Important)
    # -------------------------------------------------

    scale_pos_weight = (
        len(y_train[y_train == 0]) /
        len(y_train[y_train == 1])
    )

    print("\n⚖ scale_pos_weight:", scale_pos_weight)

    # -------------------------------------------------
    # Model (More Stable Configuration)
    # -------------------------------------------------

    print("\n🔥 Training Improved Binary Model...")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        reg_alpha=2,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # -------------------------------------------------
    # Validation Evaluation
    # -------------------------------------------------

    val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = optimize_threshold(y_val, val_probs)

    print("\n🚀 Best Threshold:", best_thresh)
    print("Best Validation F1:", best_f1)

    val_preds = (val_probs > best_thresh).astype(int)

    print("\n📈 Validation Performance:")
    print(classification_report(y_val, val_preds))
    print("Validation Accuracy:",
          accuracy_score(y_val, val_preds))

    # -------------------------------------------------
    # Test Evaluation
    # -------------------------------------------------

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > best_thresh).astype(int)

    print("\n📊 Test Performance:")
    print(classification_report(y_test, test_preds))
    print("Test Accuracy:",
          accuracy_score(y_test, test_preds))

    # -------------------------------------------------
    # NEW ADDITION → SAVE MODEL METRICS
    # -------------------------------------------------

    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    accuracy = accuracy_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    joblib.dump(metrics, "models/model_metrics.pkl")

    print("\n📊 Model Metrics Saved:")
    print(metrics)

    # -------------------------------------------------
    # Save Model + Threshold
    # -------------------------------------------------

    os.makedirs("models", exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f"models/sla_binary_stable_model_{version}.pkl"

    joblib.dump(model, model_path)
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(best_thresh, "models/failure_threshold.pkl")

    print("\n✅ Stable Binary Model Saved:", model_path)


if __name__ == "__main__":
    main()
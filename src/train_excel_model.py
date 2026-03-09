import os
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier


# -------------------------------------------------
# Project Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "merged_dataset_final.xlsx")

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------------------------------
# Sliding Time Split
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

    # Ensure test set has both classes
    if test_df["target_binary"].nunique() < 2:

        print("⚠ Test set has only one class. Adjusting...")

        class0 = val_df[val_df["target_binary"] == 0]
        class1 = val_df[val_df["target_binary"] == 1]

        extra0 = class0.sample(min(len(class0), 1000), random_state=42)
        extra1 = class1.sample(min(len(class1), 1000), random_state=42)

        extra = pd.concat([extra0, extra1])

        test_df = pd.concat([test_df, extra])
        val_df = val_df.drop(extra.index)

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

    print("🚀 Loading Excel Data...")

    df = pd.read_excel(DATA_PATH)

    print("Dataset Shape:", df.shape)

    # -------------------------------------------------
    # Target Cleaning
    # -------------------------------------------------

    print("\nCommit Status Distribution:")
    print(df["commit_status"].value_counts())

    df["target_binary"] = df["commit_status"].astype(str).str.lower().apply(
        lambda x: 0 if "ontime" in x else 1
    )

    print("\nTarget Distribution:")
    print(df["target_binary"].value_counts())

    # -------------------------------------------------
    # Timestamp Conversion
    # -------------------------------------------------

    df["IST_svc_commit_tmstp"] = pd.to_datetime(
        df["IST_svc_commit_tmstp"],
        errors="coerce",
        format="mixed"
    )

    df["last_scan"] = pd.to_datetime(
        df["last_scan"],
        errors="coerce",
        format="mixed"
    )

    df = df.dropna(subset=["IST_svc_commit_tmstp"])

    # -------------------------------------------------
    # Time Features
    # -------------------------------------------------

    df["commit_hour"] = df["IST_svc_commit_tmstp"].dt.hour
    df["commit_day"] = df["IST_svc_commit_tmstp"].dt.day
    df["commit_weekday"] = df["IST_svc_commit_tmstp"].dt.weekday

    # -------------------------------------------------
    # Logistics Feature
    # -------------------------------------------------

    df["scan_delay_hours"] = (
        df["IST_svc_commit_tmstp"] - df["last_scan"]
    ).dt.total_seconds() / 3600

    if df["scan_delay_hours"].notna().sum() > 0:
        df["scan_delay_hours"] = df["scan_delay_hours"].fillna(
            df["scan_delay_hours"].median()
        )
    else:
        df["scan_delay_hours"] = 0

    # -------------------------------------------------
    # Missing Values (Safe Handling)
    # -------------------------------------------------

    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].astype(str).fillna("Unknown")

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

    remove_cols = ["commit_status"]

    categorical_cols = [c for c in categorical_cols if c not in remove_cols]

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

    # -------------------------------------------------
    # Drop Columns (Prevent Leakage)
    # -------------------------------------------------

    drop_cols = [
        "commit_status",
        "target_binary",
        "IST_svc_commit_tmstp",
        "last_scan",
        "Trk Nos",
        "Prime Trk Nos",
        "Consignee Name",
        "Consignee Comp",
        "Dest Loc",
        "Last Scan Loc",
        "City name",
        "recp_pstl_cd"
    ]

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df["target_binary"]

    X_val = val_df.drop(columns=drop_cols, errors="ignore")
    y_val = val_df["target_binary"]

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df["target_binary"]

    # -------------------------------------------------
    # Class Imbalance Handling
    # -------------------------------------------------

    pos = len(y_train[y_train == 1])
    neg = len(y_train[y_train == 0])

    scale_pos_weight = neg / pos if pos > 0 else 1

    print("scale_pos_weight:", scale_pos_weight)

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    print("\n🔥 Training XGBoost Model...")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=450,
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
    # Validation
    # -------------------------------------------------

    val_probs = model.predict_proba(X_val)[:, 1]

    best_thresh, best_f1 = optimize_threshold(y_val, val_probs)

    print("Best Threshold:", best_thresh)
    print("Best Validation F1:", best_f1)

    # -------------------------------------------------
    # Test Performance
    # -------------------------------------------------

    test_probs = model.predict_proba(X_test)[:, 1]

    test_preds = (test_probs > best_thresh).astype(int)

    print("\n📊 Test Performance")

    print(classification_report(y_test, test_preds))

    print("Accuracy:", accuracy_score(y_test, test_preds))

    # -------------------------------------------------
    # Save Model
    # -------------------------------------------------

    version = datetime.now().strftime("%Y%m%d_%H%M")

    model_path = os.path.join(
        MODEL_DIR,
        f"shipment_model_{version}.pkl"
    )

    joblib.dump(model, model_path)

    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders_excel.pkl"))

    joblib.dump(best_thresh, os.path.join(MODEL_DIR, "failure_threshold_excel.pkl"))

    print("\n✅ Model Saved:", model_path)


if __name__ == "__main__":
    main()
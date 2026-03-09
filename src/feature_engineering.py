import pandas as pd


def create_time_features(df):
    df = df.copy()

    if "IST_svc_commit_tmstp" in df.columns:
        df["IST_svc_commit_tmstp"] = pd.to_datetime(
            df["IST_svc_commit_tmstp"],
            errors="coerce"
        )

        df["commit_hour"] = df["IST_svc_commit_tmstp"].dt.hour
        df["commit_day"] = df["IST_svc_commit_tmstp"].dt.day
        df["commit_month"] = df["IST_svc_commit_tmstp"].dt.month
        df["commit_weekday"] = df["IST_svc_commit_tmstp"].dt.weekday

    return df
import pandas as pd


def load_data(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def clean_target(df):
    df = df.copy()

    # Normalize commit_status safely
    df["commit_status"] = (
        df["commit_status"]
        .astype(str)
        .str.strip()
        .str.replace("_", " ", regex=False)   # FIXED
        .str.upper()
    )

    mapping = {
        "ONTIME": 0,
        "COMMIT FAIL": 1,
        "POD COMMIT FAIL": 2
    }

    df["target"] = df["commit_status"].map(mapping)

    # Drop only rows where mapping failed
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    print("\n🎯 Target Distribution:")
    print(df["target"].value_counts())

    return df


def drop_irrelevant_columns(df):
    df = df.copy()

    cols_to_drop = [
        "Prime Trk Nos",
        "Emp Nos",
        "Consignee Comp",
        "Consignee Name",
        "Map"
    ]

    return df.drop(columns=cols_to_drop, errors="ignore")
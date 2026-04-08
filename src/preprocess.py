import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df):
    df = df.copy()

    # Handle missing values
    df.fillna(method="ffill", inplace=True)

    return df


def encode_data(df):
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def scale_data(df):
    df = df.copy()
    scaler = StandardScaler()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

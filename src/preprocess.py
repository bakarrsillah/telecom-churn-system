import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------
# LOAD DATA
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df

# -------------------------
# CLEAN DATA
# -------------------------
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    return df

# -------------------------
# ENCODE CATEGORICAL DATA
# -------------------------
def encode_data(df):
    le = LabelEncoder()

    if 'location' in df.columns:
        df['location'] = le.fit_transform(df['location'])

    return df

# -------------------------
# FEATURE SCALING
# -------------------------
def scale_data(df):
    scaler = StandardScaler()

    feature_cols = df.drop(columns=['customer_id', 'churn']).columns

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df

# -------------------------
# FULL PIPELINE
# -------------------------
def preprocess_pipeline(input_path, output_path):
    df = load_data(input_path)
    df = clean_data(df)
    df = encode_data(df)
    df = scale_data(df)

    df.to_csv(output_path, index=False)

    print("✅ Data preprocessing complete → saved to", output_path)

    return df
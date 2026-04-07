from src.preprocess import preprocess_pipeline
from src.features import create_features
from src.train_model import train_model
import pandas as pd

# -------------------------
# STEP 1: PREPROCESS
# -------------------------
df = preprocess_pipeline(
    input_path="data/raw.csv",
    output_path="data/processed.csv"
)

# -------------------------
# STEP 2: FEATURE ENGINEERING
# -------------------------
df = create_features(df)
df.to_csv("data/processed.csv", index=False)

print("✅ Feature engineering complete")

# -------------------------
# STEP 3: TRAIN MODEL
# -------------------------
model = train_model("data/processed.csv")
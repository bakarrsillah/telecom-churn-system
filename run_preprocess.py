from src.preprocess import preprocess_pipeline
from src.features import create_features
import pandas as pd

# Run preprocessing
df = preprocess_pipeline(
    input_path="data/raw.csv",
    output_path="data/processed.csv"
)

# Add features
df = create_features(df)

# Save updated dataset
df.to_csv("data/processed.csv", index=False)

print("✅ Feature engineering complete")
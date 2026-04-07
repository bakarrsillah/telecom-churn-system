import pandas as pd
from src.predict import predict_churn, assign_risk, recommend_action

# Load processed data
df = pd.read_csv("data/processed.csv")

# Remove target column
X = df.drop(columns=["churn"])

# Predict probabilities
probs = predict_churn(X)

# Add results
df["churn_probability"] = probs
df["risk_level"] = df["churn_probability"].apply(assign_risk)
df["action"] = df["risk_level"].apply(recommend_action)

# Show sample
print(df[["customer_id", "churn_probability", "risk_level", "action"]].head())

# Save results
df.to_csv("data/predictions.csv", index=False)

print("✅ Predictions saved to data/predictions.csv")